import asyncio
import base64
from io import BytesIO
from typing import Optional, AsyncIterator, Callable, Awaitable, Any
from pathlib import Path

from PIL import Image


# Pre-import transformers to avoid race conditions during parallel initialization.
# The transformers library uses lazy loading which is not thread-safe when multiple
# threads try to import different symbols simultaneously.
def _preload_transformers():
    try:
        import transformers

        # Trigger lazy loading of commonly used Auto classes
        _ = transformers.AutoProcessor
        _ = transformers.AutoModelForVision2Seq
        _ = transformers.AutoFeatureExtractor
        _ = transformers.AutoModel
    except ImportError:
        pass


_preload_transformers()


# Monkey-patch torch.repeat_interleave for ROCm compatibility
# ROCm 6.3 has a bug where repeat_interleave with tensor repeats fails with HIP error
def _patch_repeat_interleave_for_rocm():
    import torch

    if not torch.cuda.is_available():
        return

    # Check if we're on ROCm (HIP)
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        return

    _original_repeat_interleave = torch.repeat_interleave

    def _patched_repeat_interleave(input, repeats, dim=None, *, output_size=None):
        # If repeats is a tensor on CUDA, move to CPU for the operation
        if isinstance(repeats, torch.Tensor) and repeats.is_cuda:
            input_device = input.device if isinstance(input, torch.Tensor) else None
            result = _original_repeat_interleave(
                input.cpu()
                if isinstance(input, torch.Tensor) and input.is_cuda
                else input,
                repeats.cpu(),
                dim=dim,
                output_size=output_size,
            )
            if input_device is not None and input_device.type == "cuda":
                return result.to(input_device)
            return result
        return _original_repeat_interleave(
            input, repeats, dim=dim, output_size=output_size
        )

    torch.repeat_interleave = _patched_repeat_interleave
    print("[*] Patched torch.repeat_interleave for ROCm compatibility")


_patch_repeat_interleave_for_rocm()

from .config import CharacterConfig, SessionConfig, DATA_DIR, CONVERSATION_HISTORY_LIMIT
from .vm_manager import VMManager
from .memory import MemoryManager, ConversationHistory
from .tts import TTSEngine
from .tools import ToolExecutor, ToolResult, parse_tool_call


SYSTEM_PROMPT_TEMPLATE = """You are an AI agent controlling a virtual machine. You can see the screen and interact with it using various tools.

{character_system_prompt}

## Character Personality
{personality}

## Available Tools
Use tools by wrapping them in <tool></tool> tags with YAML content.

### Tools:
1. wait - Wait for specified seconds
   <tool>
   name: wait
   seconds: 2
   </tool>

2. input - Press a key or key combination
   <tool>
   name: input
   key: Enter
   </tool>
   <tool>
   name: input
   key: Ctrl+C
   </tool>

3. text - Type text (note: follows current keyboard layout)
   <tool>
   name: text
   text: Hello World
   </tool>

4. cursor-tp - Move cursor to absolute position (screen is {width}x{height})
   <tool>
   name: cursor-tp
   x: 640
   y: 360
   </tool>

5. cursor-move - Move cursor relative to current position
   <tool>
   name: cursor-move
   dx: 100
   dy: -50
   </tool>

6. click - Click mouse button at current position
   <tool>
   name: click
   button: left
   </tool>

7. click-on - Hold mouse button down
   <tool>
   name: click-on
   button: left
   </tool>

8. click-off - Release mouse button
   <tool>
   name: click-off
   button: left
   </tool>

9. speak - Say something to the user
   <tool>
   name: speak
   text: I found the file you requested.
   </tool>

10. memory - Search past conversation memories
    <tool>
    name: memory
    query: previous file operations
    </tool>

11. todo - Manage your task list
    <tool>
    name: todo
    action: read
    </tool>
    <tool>
    name: todo
    action: write
    content: |
      - [ ] Task 1
      - [x] Task 2 (done)
    </tool>

12. end - Complete the current task
    <tool>
    name: end
    reason: Task completed successfully
    </tool>

## Rules
- You can only use ONE tool per response
- Always observe the result before proceeding
- Use the speak tool to communicate with the user
- Track your progress using the todo tool
- When the task is complete, use the end tool

## Current Task
{task}

## Your Todo List
{todo_content}
"""


class LLMProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._is_vlm = False
        self._initialized = False
        self._device = "cpu"

    async def initialize(self):
        if self._initialized:
            return

        print(f"[*] Loading model: {self.model_name}")
        print(
            "[*] This may take a while on first run (downloading from Hugging Face)..."
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)
        self._initialized = True
        if self._model is not None:
            print(f"[+] Model loaded successfully on {self._device}")

    def _init_sync(self):
        import torch

        models_dir = DATA_DIR / "llm_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        vlm_keywords = [
            "qwen3-vl",
            "llava",
            "florence",
            "paligemma",
            "idefics",
            "cogvlm",
            "internvl",
        ]
        self._is_vlm = any(kw in self.model_name.lower() for kw in vlm_keywords)

        try:
            if self._is_vlm:
                self._load_vlm(models_dir, dtype)
            else:
                self._load_text_model(models_dir, dtype)
        except Exception as e:
            print(f"[!] Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            self._model = None
            self._processor = None

    def _load_vlm(self, models_dir: Path, dtype):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print("[*] Detected Vision-Language Model")

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=str(models_dir), trust_remote_code=True
        )

        if "qwen3-vl" in self.model_name.lower():
            from transformers import Qwen3VLForConditionalGeneration

            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=str(models_dir),
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                cache_dir=str(models_dir),
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )

        if self._device == "cpu" and hasattr(self._model, "to"):
            self._model = self._model.to(self._device)

    def _load_text_model(self, models_dir: Path, dtype):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[*] Loading text-only model (no vision capability)")

        self._processor = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=str(models_dir), trust_remote_code=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(models_dir),
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )

        if self._device == "cpu":
            self._model = self._model.to(self._device)

    async def generate(
        self,
        messages: list[dict],
        image: Optional[Image.Image] = None,
        max_tokens: int = 1024,
    ) -> str:
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, messages, image, max_tokens
        )

    def _generate_sync(
        self, messages: list[dict], image: Optional[Image.Image], max_tokens: int
    ) -> str:
        import torch

        if not self._processor or not self._model:
            raise RuntimeError("Model not initialized")

        if self._is_vlm and image:
            return self._generate_vlm(messages, image, max_tokens)
        else:
            return self._generate_text(messages, max_tokens)

    def _generate_vlm(
        self, messages: list[dict], image: Image.Image, max_tokens: int
    ) -> str:
        import torch

        prompt = self._format_messages(messages)

        if "qwen3-vl" in self.model_name.lower():
            from qwen_vl_utils import process_vision_info

            qwen_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(qwen_messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7
            )

        input_len = inputs.get("input_ids", inputs.get("input_token_ids", [[]])).shape[
            1
        ]
        response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        return response.strip()

    def _generate_text(self, messages: list[dict], max_tokens: int) -> str:
        import torch

        prompt = self._format_messages(messages)

        inputs = self._processor(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._processor.eos_token_id,
            )

        response = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()

    def _format_messages(self, messages: list[dict]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"System: {content}\n")
            elif role == "user":
                formatted.append(f"User: {content}\n")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}\n")

        formatted.append("Assistant: ")
        return "".join(formatted)


class AIAgent:
    def __init__(self, session_config: SessionConfig):
        self.config = session_config
        self.vm: Optional[VMManager] = None
        self.llm: Optional[LLMProvider] = None
        self.tts: Optional[TTSEngine] = None
        self.memory: Optional[MemoryManager] = None
        self.history: Optional[ConversationHistory] = None
        self.tools: Optional[ToolExecutor] = None

        self._running = False
        self._todo_path: Optional[Path] = None
        self._on_speak_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_tool_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None
        self._on_screen_callback: Optional[Callable[[str], Awaitable[None]]] = None

    def set_callbacks(
        self,
        on_speak: Optional[Callable[[str], Awaitable[None]]] = None,
        on_tool: Optional[Callable[[str, dict], Awaitable[None]]] = None,
        on_screen: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._on_speak_callback = on_speak
        self._on_tool_callback = on_tool
        self._on_screen_callback = on_screen

    async def initialize(self):
        self.llm = LLMProvider(self.config.character.llm_model)
        self.tts = TTSEngine(
            self.config.character.voice_file, self.config.character.voice_text
        )

        session_id = f"session_{id(self)}"
        self.memory = MemoryManager(session_id)

        self.history = ConversationHistory(max_recent=CONVERSATION_HISTORY_LIMIT)
        self.history.set_memory(self.memory)

        self._todo_path = DATA_DIR / "temp" / f"{session_id}_todo.md"
        self._todo_path.parent.mkdir(parents=True, exist_ok=True)
        self._todo_path.write_text("")

        # Initialize LLM first (needs clean GPU state), then others
        # TTS has meta tensor bugs that can corrupt GPU state if run in parallel
        await self.llm.initialize()
        await asyncio.gather(self.tts.initialize(), self.memory.initialize())

    async def start_vm(self):
        from .config import VMConfig

        self.vm = VMManager(self.config.vm)
        await self.vm.start()

        self.tools = ToolExecutor(
            vm=self.vm,
            memory=self.memory,
            tts=self.tts,
            on_speak=self._handle_speak,
            todo_path=str(self._todo_path),
        )

    async def stop_vm(self):
        if self.vm:
            await self.vm.stop()
            self.vm = None

    async def _handle_speak(self, text: str):
        if self._on_speak_callback:
            await self._on_speak_callback(text)

    def _get_todo_content(self) -> str:
        if self._todo_path and self._todo_path.exists():
            return self._todo_path.read_text()
        return "(empty)"

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            character_system_prompt=self.config.character.system_prompt,
            personality=self.config.character.personality,
            width=self.config.vm.width,
            height=self.config.vm.height,
            task=self.config.task,
            todo_content=self._get_todo_content(),
        )

    async def _capture_screen(self) -> tuple[Optional[Image.Image], str]:
        if not self.vm:
            return None, ""

        screen = await self.vm.capture_screen()

        buffer = BytesIO()
        screen.save(buffer, format="PNG")
        b64_image = base64.b64encode(buffer.getvalue()).decode()

        if self._on_screen_callback:
            await self._on_screen_callback(b64_image)

        return screen, b64_image

    async def run_step(self) -> tuple[str, Optional[ToolResult]]:
        if not self.llm or not self.history or not self.tools:
            raise RuntimeError("Agent not initialized")

        print("[*] Capturing screen...")
        screen_image, screen_b64 = await self._capture_screen()
        if screen_image:
            print(f"[+] Screen captured ({screen_image.width}x{screen_image.height})")
        else:
            print("[!] Failed to capture screen")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
        ]

        for msg in self.history.get_recent():
            messages.append(msg)

        messages.append(
            {
                "role": "user",
                "content": "[Current VM screen is attached. Analyze and take action.]",
            }
        )

        print("[*] Generating LLM response (this may take a while on CPU)...")
        import time

        start_time = time.time()
        response = await self.llm.generate(messages, image=screen_image)
        elapsed = time.time() - start_time
        print(f"[+] LLM response generated in {elapsed:.1f}s")

        tool_call = parse_tool_call(response)
        tool_result = None

        if tool_call:
            tool_name, params = tool_call

            if self._on_tool_callback:
                await self._on_tool_callback(tool_name, params)

            tool_result = await self.tools.execute(tool_name, params)

            await self.history.add_message(
                "assistant",
                response,
                tool_use={
                    "name": tool_name,
                    "params": params,
                    "result": tool_result.message,
                },
            )
        else:
            await self.history.add_message("assistant", response)

        return response, tool_result

    async def run(self) -> AsyncIterator[tuple[str, Optional[ToolResult]]]:
        self._running = True
        print("[*] Agent loop started")

        step = 0
        while self._running:
            step += 1
            print(f"\n{'=' * 50}")
            print(f"[*] Agent step {step}")
            response, result = await self.run_step()
            yield response, result

            if result and not result.should_continue:
                print(f"[*] Agent stopping: {result.message}")
                break

            await asyncio.sleep(0.5)

    def stop(self):
        self._running = False
