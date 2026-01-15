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


class GPUMemoryManager:
    GPU_MEMORY_THRESHOLD = 0.90

    @staticmethod
    def get_gpu_memory_info() -> dict:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            free = total - reserved

            return {
                "available": True,
                "allocated_mb": allocated / (1024 * 1024),
                "reserved_mb": reserved / (1024 * 1024),
                "total_mb": total / (1024 * 1024),
                "free_mb": free / (1024 * 1024),
                "usage_ratio": reserved / total if total > 0 else 0,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    @staticmethod
    def synchronize():
        """Synchronize GPU to prevent race conditions and ensure all operations complete."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    @staticmethod
    def clear_cache():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all GPU ops before clearing
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                print("[*] GPU cache cleared")
        except Exception as e:
            print(f"[!] Failed to clear GPU cache: {e}")

    @staticmethod
    def reset_device():
        """Reset CUDA device after critical errors to recover from corrupted state."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                # Reset accumulated memory stats
                torch.cuda.reset_accumulated_memory_stats()
                import gc

                gc.collect()
                print("[*] GPU device reset completed")
        except Exception as e:
            print(f"[!] Failed to reset GPU device: {e}")

    @staticmethod
    def check_memory_and_cleanup() -> bool:
        info = GPUMemoryManager.get_gpu_memory_info()
        if not info.get("available"):
            return True

        usage = info.get("usage_ratio", 0)
        if usage > GPUMemoryManager.GPU_MEMORY_THRESHOLD:
            print(f"[!] GPU memory usage high ({usage:.1%}), clearing cache...")
            GPUMemoryManager.clear_cache()

            new_info = GPUMemoryManager.get_gpu_memory_info()
            new_usage = new_info.get("usage_ratio", 0)
            print(f"[*] GPU memory after cleanup: {new_usage:.1%}")

            return new_usage < GPUMemoryManager.GPU_MEMORY_THRESHOLD

        return True


from .config import CharacterConfig, SessionConfig, DATA_DIR, CONVERSATION_HISTORY_LIMIT
from .vm_manager import VMManager
from .memory import MemoryManager, ConversationHistory
from .tts import TTSEngine
from .tools import ToolExecutor, ToolResult, parse_tool_calls


SYSTEM_PROMPT_TEMPLATE = """You are an AI agent controlling a virtual machine.

{character_system_prompt}

## Personality
{personality}

---

# CRITICAL: ONE ACTION PER TURN

You operate in a **turn-based loop**:
1. You SEE the screen
2. You THINK about what to do  
3. You output ONE action
4. You WAIT for the result (next screenshot)
5. Repeat

**NEVER chain multiple actions. You will see the result after each action.**

---

# MANDATORY RESPONSE FORMAT

Every response MUST follow this exact structure:

```
<thinking>
1. OBSERVATION: What do I see? Where is the cursor now? (x, y coordinates)
2. GOAL: What am I trying to accomplish in this step?
3. DECISION: What single action should I take?
4. VERIFY (for clicks): Is cursor EXACTLY on target? Yes/No
</thinking>

<tool>
name: speak
text: [Explain what you're about to do]
</tool>

<tool>
name: [ONE action tool]
[parameters]
</tool>
```

## Rules
- `<thinking>` block is MANDATORY before any action
- `speak` tool is MANDATORY - never skip it
- Only ONE action tool after speak (cursor-move OR click OR text, etc.)
- After outputting, STOP and wait for the next screenshot

---

# TOOLS

## Communication (ALWAYS REQUIRED)
**speak** - Tell user what you're doing
```yaml
<tool>
name: speak
text: I'll click on the Start menu.
</tool>
```

## Mouse Control
**cursor-move** - Move cursor relative to current position
```yaml
<tool>
name: cursor-move
dx: 100
dy: -50
</tool>
```

**click** - Click at current cursor position
```yaml
<tool>
name: click
button: left
</tool>
```

**click-on** / **click-off** - Hold/release mouse button (for dragging)

## Keyboard
**input** - Press key or combo
```yaml
<tool>
name: input
key: Enter
</tool>
```

**text** - Type text
```yaml
<tool>
name: text
text: Hello World
</tool>
```

## Other
**wait** - Wait seconds (use when screen is loading)
**memory** - Search past conversations
**todo** - Manage task list (action: read/write)
**end** - Complete task (use with speak)

---

# CLICKING PROTOCOL (CRITICAL)

Before EVERY click, you MUST verify in your `<thinking>` block:

1. Look at the cursor position in the screenshot
2. Look at where you want to click
3. Ask: "Is the cursor EXACTLY on the target?"
   - **NO** → Use `cursor-move` to adjust. Do NOT click yet.
   - **YES** → Safe to use `click`

**COMMON MISTAKE**: Using `cursor-move` then immediately `click` without seeing the result.
**CORRECT**: `cursor-move` → wait for screenshot → verify position → then `click` (next turn)

---

# EXAMPLES

## Example 1: Moving cursor (NOT clicking yet)
```
<thinking>
1. OBSERVATION: Cursor is at top-left (~50, 50). I see a "Firefox" icon at (~100, 400).
2. GOAL: Move cursor to the Firefox icon.
3. DECISION: Use cursor-move to get closer. Will click AFTER I verify position.
4. VERIFY: N/A - just moving, not clicking.
</thinking>

<tool>
name: speak
text: Moving cursor toward the Firefox icon.
</tool>

<tool>
name: cursor-move
dx: 50
dy: 350
</tool>
```

## Example 2: Clicking after verification
```
<thinking>
1. OBSERVATION: Cursor is now at (~100, 400), directly on the Firefox icon center.
2. GOAL: Click to open Firefox.
3. DECISION: Cursor is on target, safe to click.
4. VERIFY: YES - cursor is exactly on the Firefox icon.
</thinking>

<tool>
name: speak
text: Clicking on Firefox to open the browser.
</tool>

<tool>
name: click
button: left
</tool>
```

## Example 3: Cursor not on target - DO NOT CLICK
```
<thinking>
1. OBSERVATION: Cursor is at (~80, 380), slightly off from the icon center at (~100, 400).
2. GOAL: Click Firefox icon.
3. DECISION: Cursor is NOT on target. Must adjust first.
4. VERIFY: NO - cursor is 20px off. Need to move.
</thinking>

<tool>
name: speak
text: Adjusting cursor position to center on the icon.
</tool>

<tool>
name: cursor-move
dx: 20
dy: 20
</tool>
```

---

# CURRENT TASK
{task}

# YOUR TODO LIST
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

        # Get LLM device from gpu_utils
        try:
            from .gpu_utils import get_llm_device, print_gpu_info

            print_gpu_info()
            self._device = get_llm_device()
        except ImportError:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        use_gpu = self._device != "cpu"
        dtype = torch.float16 if use_gpu else torch.float32

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

        # Determine device_map based on LLM device
        if self._device == "cpu":
            device_map = None
        elif ":" in self._device:
            # Specific GPU like "cuda:0" - use that device
            device_map = {"": self._device}
        else:
            device_map = "auto"

        if "qwen3-vl" in self.model_name.lower():
            from transformers import Qwen3VLForConditionalGeneration

            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=str(models_dir),
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                cache_dir=str(models_dir),
                torch_dtype=dtype,
                device_map=device_map,
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
        video_frames: Optional[list[Image.Image]] = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate response from LLM.

        Args:
            messages: Chat messages
            image: Single image (legacy support)
            video_frames: List of video frames (preferred for temporal context)
            max_tokens: Maximum tokens to generate
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, messages, image, video_frames, max_tokens
        )

    def _generate_sync(
        self,
        messages: list[dict],
        image: Optional[Image.Image],
        video_frames: Optional[list[Image.Image]],
        max_tokens: int,
    ) -> str:
        import torch

        if not self._processor or not self._model:
            raise RuntimeError("Model not initialized")

        if not GPUMemoryManager.check_memory_and_cleanup():
            print("[!] GPU memory critically low, attempting reduced generation")

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if self._is_vlm and (image or video_frames):
                    return self._generate_vlm(messages, image, video_frames, max_tokens)
                else:
                    return self._generate_text(messages, max_tokens)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                error_str = str(e).lower()
                is_oom = "out of memory" in error_str or isinstance(
                    e, torch.cuda.OutOfMemoryError
                )
                # Detect CUDA/HIP critical errors that require device reset
                is_cuda_error = any(
                    x in error_str
                    for x in [
                        "cuda error",
                        "hip error",
                        "device-side assert",
                        "illegal memory access",
                        "cublas",
                        "cudnn",
                    ]
                )

                if is_oom:
                    print(f"[!] GPU OOM on attempt {attempt + 1}/{max_retries + 1}")
                    GPUMemoryManager.clear_cache()

                    if attempt == max_retries:
                        print("[!] Max OOM retries reached, returning error message")
                        return "<tool>\nname: speak\ntext: I encountered a memory issue. Please try restarting me.\n</tool>"

                    max_tokens = max(256, max_tokens // 2)
                    print(f"[*] Reducing max_tokens to {max_tokens}")
                elif is_cuda_error:
                    print(
                        f"[!] CUDA/HIP error on attempt {attempt + 1}/{max_retries + 1}: {e}"
                    )
                    # Reset device to recover from corrupted GPU state
                    GPUMemoryManager.reset_device()

                    if attempt == max_retries:
                        print(
                            "[!] Max CUDA error retries reached, returning error message"
                        )
                        return "<tool>\nname: speak\ntext: I encountered a GPU error. Please try restarting me.\n</tool>"

                    print("[*] GPU device reset, retrying...")
                else:
                    raise

    def _generate_vlm(
        self,
        messages: list[dict],
        image: Optional[Image.Image],
        video_frames: Optional[list[Image.Image]],
        max_tokens: int,
    ) -> str:
        import torch
        import gc

        if "qwen3-vl" in self.model_name.lower():
            from qwen_vl_utils import process_vision_info

            qwen_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    qwen_messages.append({"role": "system", "content": content})
                elif role == "assistant":
                    qwen_messages.append({"role": "assistant", "content": content})
                elif role == "user":
                    qwen_messages.append({"role": "user", "content": content})

            # Build vision content - prefer video frames over single image
            if video_frames and len(video_frames) > 0:
                # Use video frames for temporal context
                vision_content = [
                    {"type": "video", "video": video_frames, "fps": 1.0},
                    {
                        "type": "text",
                        "text": "These are the recent screen frames (oldest to newest). Analyze the progression and take the next action based on your task. Do NOT repeat the same action.",
                    },
                ]
            elif image:
                # Fallback to single image
                vision_content = [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "This is the current screen. Take action based on your task and previous actions. Do NOT repeat the same action.",
                    },
                ]
            else:
                raise ValueError("No image or video frames provided")

            qwen_messages.append(
                {
                    "role": "user",
                    "content": vision_content,
                }
            )

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
            prompt = self._format_messages(messages)
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Synchronize before generation to ensure clean GPU state
        GPUMemoryManager.synchronize()

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7
                )

            # Synchronize after generation to ensure completion
            GPUMemoryManager.synchronize()

            input_len = inputs.get(
                "input_ids", inputs.get("input_token_ids", [[]])
            ).shape[1]
            response = self._processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

            return response.strip()
        finally:
            # Explicitly clean up tensors to prevent memory fragmentation
            del inputs
            gc.collect()
            GPUMemoryManager.synchronize()

    def _generate_text(self, messages: list[dict], max_tokens: int) -> str:
        import torch
        import gc

        prompt = self._format_messages(messages)

        inputs = self._processor(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Synchronize before generation to ensure clean GPU state
        GPUMemoryManager.synchronize()

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self._processor.eos_token_id,
                )

            # Synchronize after generation to ensure completion
            GPUMemoryManager.synchronize()

            response = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return response.strip()
        finally:
            # Explicitly clean up tensors to prevent memory fragmentation
            del inputs
            gc.collect()
            GPUMemoryManager.synchronize()

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
    # Number of frames to capture for video context
    FRAME_BUFFER_SIZE = 6
    # Interval between frame captures (seconds)
    FRAME_CAPTURE_INTERVAL = 0.5

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

        # Frame buffer for video context
        self._frame_buffer: list[Image.Image] = []

    def set_callbacks(
        self,
        on_speak: Optional[Callable[[str], Awaitable[None]]] = None,
        on_tool: Optional[Callable[[str, dict], Awaitable[None]]] = None,
        on_screen: Optional[Callable[[str], Awaitable[None]]] = None,
        on_audio: Optional[Callable[[bytes], Awaitable[None]]] = None,
    ):
        self._on_speak_callback = on_speak
        self._on_tool_callback = on_tool
        self._on_screen_callback = on_screen
        self._on_audio_callback = on_audio

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

    async def _process_audio_queue(self):
        """Monitor TTS queue and send audio to frontend."""
        while self._running:
            try:
                if self.tts:
                    audio_data = await self.tts.get_next_audio()
                    if audio_data and self._on_audio_callback:
                        await self._on_audio_callback(audio_data)

                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[!] Error in audio loop: {e}")
                await asyncio.sleep(1)

    async def _video_recorder_loop(self):
        """
        Background loop that continuously captures frames.
        This provides a history of what happened since the last LLM action.
        """
        print("[*] Video recorder started")
        try:
            while self._running:
                try:
                    # Capture frame without triggering callbacks (to avoid spamming UI)
                    # We only send to UI when run_step captures or explicitly requested
                    if self.vm:
                        screen = await self.vm.capture_screen()
                        self._frame_buffer.append(screen)

                        # Limit buffer size (30 seconds at 2fps)
                        if len(self._frame_buffer) > 60:
                            self._frame_buffer.pop(0)

                except Exception as e:
                    # Don't crash the loop
                    pass

                await asyncio.sleep(self.FRAME_CAPTURE_INTERVAL)
        except asyncio.CancelledError:
            pass
        print("[*] Video recorder stopped")

    async def run_step(self) -> tuple[str, Optional[ToolResult]]:
        if not self.llm or not self.history or not self.tools:
            raise RuntimeError("Agent not initialized")

        # 1. Get accumulated video frames (history since last step)
        video_frames = list(self._frame_buffer)
        self._frame_buffer.clear()  # Clear buffer for next step

        # 2. Capture current state (the "now" frame)
        print("[*] Capturing current screen...")
        screen_image, screen_b64 = await self._capture_screen()

        if screen_image:
            video_frames.append(screen_image)
            print(f"[+] Screen captured ({screen_image.width}x{screen_image.height})")
            print(f"[+] Video context: {len(video_frames)} frames")
        else:
            print("[!] Failed to capture screen")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
        ]

        history_msgs = self.history.get_recent()
        print(f"[DEBUG] History has {len(history_msgs)} messages")
        for msg in history_msgs:
            messages.append(msg)

        messages.append(
            {
                "role": "user",
                "content": "[Video frames of VM screen are attached. Analyze the progression and take the next action.]",
            }
        )

        print(f"[*] Generating LLM response (on {self.llm._device})...")
        import time

        start_time = time.time()
        # Pass video frames for temporal context
        response = await self.llm.generate(
            messages, image=screen_image, video_frames=video_frames
        )
        elapsed = time.time() - start_time
        print(f"[+] LLM response generated in {elapsed:.1f}s")
        print(f"[DEBUG] LLM response:\n{response[:500]}...")

        tool_calls = parse_tool_calls(response)
        print(f"[DEBUG] Parsed {len(tool_calls)} tool(s): {[t[0] for t in tool_calls]}")
        tool_result = None
        tool_uses = []

        for tool_name, params in tool_calls:
            print(f"[*] Executing tool: {tool_name} with params: {params}")
            if self._on_tool_callback:
                await self._on_tool_callback(tool_name, params)

            tool_result = await self.tools.execute(tool_name, params)
            print(f"[+] Tool result: {tool_result.message}")
            tool_uses.append(
                {
                    "name": tool_name,
                    "params": params,
                    "result": tool_result.message,
                }
            )

            if not tool_result.should_continue:
                break

        if tool_uses:
            await self.history.add_message(
                "assistant",
                response,
                tool_use=tool_uses[0] if len(tool_uses) == 1 else tool_uses,
            )
        else:
            await self.history.add_message("assistant", response)

        return response, tool_result

    async def run(self) -> AsyncIterator[tuple[str, Optional[ToolResult]]]:
        self._running = True

        # Start background tasks
        audio_task = asyncio.create_task(self._process_audio_queue())
        video_task = asyncio.create_task(self._video_recorder_loop())

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

            # Small delay to prevent tight loop if LLM is too fast
            await asyncio.sleep(0.5)

        # Cleanup tasks
        audio_task.cancel()
        video_task.cancel()
        try:
            await asyncio.gather(audio_task, video_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    def stop(self):
        self._running = False
