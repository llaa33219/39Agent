"""
TTS Engine using CosyVoice3.

Supports:
- Zero-shot voice cloning from reference audio
- Multi-GPU setup (uses TTS device from gpu_utils)
- Streaming audio generation
"""

import asyncio
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DATA_DIR


# CosyVoice paths
COSYVOICE_DIR = DATA_DIR / "cosyvoice"
COSYVOICE_REPO = COSYVOICE_DIR / "CosyVoice"
COSYVOICE_MODEL_DIR = COSYVOICE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"
MATCHA_TTS_PATH = COSYVOICE_REPO / "third_party" / "Matcha-TTS"


def _setup_cosyvoice_paths():
    """Add CosyVoice paths to sys.path if not already present."""
    paths_to_add = [
        str(COSYVOICE_REPO),
        str(MATCHA_TTS_PATH),
    ]
    for path in paths_to_add:
        if path not in sys.path and Path(path).exists():
            sys.path.insert(0, path)


class TTSEngine:
    """
    TTS Engine using CosyVoice3 for high-quality voice synthesis.

    Supports zero-shot voice cloning when voice_file is provided.
    """

    # Default sample rate for CosyVoice3
    DEFAULT_SAMPLE_RATE = 24000

    def __init__(
        self,
        voice_file: Optional[str] = None,
        voice_text: Optional[str] = None,
    ):
        """
        Initialize TTS Engine.

        Args:
            voice_file: Path to reference audio file for voice cloning (3-15 seconds)
            voice_text: Transcript of the reference audio
        """
        self.voice_file = voice_file
        self.voice_text = voice_text
        self._model = None
        self._initialized = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._sample_rate = self.DEFAULT_SAMPLE_RATE
        self._device = "cpu"
        self._available = False

    async def initialize(self):
        """Initialize the TTS model asynchronously."""
        if self._initialized:
            return

        print("[*] Loading TTS model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)
        self._initialized = True

        if self._available:
            print(f"[+] TTS model loaded successfully on {self._device}")
        else:
            print("[!] TTS model not available. Voice output disabled.")

    def _init_sync(self):
        """Synchronous initialization."""
        # Suppress ONNX Runtime CUDA provider errors when CUDA libs are not available
        # (e.g., libcublasLt.so.12 missing). ONNX will fallback to CPU automatically.
        try:
            import onnxruntime as ort

            ort.set_default_logger_severity(4)  # FATAL only - suppress CUDA lib errors
        except ImportError:
            pass

        # Get TTS device from gpu_utils
        try:
            from .gpu_utils import get_tts_device, print_gpu_info

            self._device = get_tts_device()
        except ImportError:
            self._device = "cpu"

        # Check if CosyVoice is installed
        if not COSYVOICE_MODEL_DIR.exists():
            print(f"[!] CosyVoice model not found at {COSYVOICE_MODEL_DIR}")
            print("[!] Run: python scripts/install_cosyvoice.py")
            self._available = False
            return

        # Setup paths
        _setup_cosyvoice_paths()

        try:
            # Import CosyVoice
            from cosyvoice.cli.cosyvoice import AutoModel

            # Try GPU first, fallback to CPU on failure
            if not self._try_load_model(AutoModel, use_gpu=self._device != "cpu"):
                if self._device != "cpu":
                    print("[!] GPU load failed, falling back to CPU...")
                    self._device = "cpu"
                    self._try_load_model(AutoModel, use_gpu=False)

            if self._available:
                # Pre-register voice if provided
                if self.voice_file and Path(self.voice_file).exists():
                    self._register_voice()

        except ImportError as e:
            print(f"[!] CosyVoice not available: {e}")
            print("[!] Run: python scripts/install_cosyvoice.py")
            self._available = False
        except Exception as e:
            print(f"[!] Failed to load CosyVoice model: {e}")
            import traceback

            traceback.print_exc()
            self._available = False

    def _try_load_model(self, AutoModel, use_gpu: bool) -> bool:
        """
        Try to load the model with given settings.

        Returns True on success, False on failure.
        """
        try:
            device_name = self._device if use_gpu else "cpu"
            print(f"[*] Loading CosyVoice3 model on {device_name}...")

            # Set CUDA device if using GPU
            if use_gpu:
                import torch
                import os

                # Extract device index from "cuda:X"
                device_idx = (
                    int(self._device.split(":")[-1]) if ":" in self._device else 0
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
                torch.cuda.set_device(
                    0
                )  # After CUDA_VISIBLE_DEVICES, it becomes device 0

            # Load model with appropriate settings
            # Note: CosyVoice3 doesn't support load_jit (only CosyVoice/CosyVoice2 do)
            self._model = AutoModel(
                model_dir=str(COSYVOICE_MODEL_DIR),
                load_trt=False,  # TensorRT requires extra setup
                load_vllm=False,  # vLLM requires extra setup
                fp16=use_gpu,  # Half precision for GPU
            )

            self._sample_rate = getattr(
                self._model, "sample_rate", self.DEFAULT_SAMPLE_RATE
            )
            self._available = True
            return True

        except Exception as e:
            error_msg = str(e).lower()
            # Check for GPU-related errors (ROCm, CUDA, compatibility issues)
            gpu_error_keywords = [
                "rocm",
                "cuda",
                "gpu",
                "device",
                "hip",
                "out of memory",
                "oom",
                "incompatible",
                "not available",
                "not supported",
                "no kernel",
            ]
            is_gpu_error = any(kw in error_msg for kw in gpu_error_keywords)

            if use_gpu and is_gpu_error:
                print(f"[!] GPU error: {e}")
                self._model = None
                self._available = False
                return False
            else:
                # Non-GPU error or CPU mode failed - raise to outer handler
                raise

    def _register_voice(self):
        """Register a reference voice for zero-shot cloning."""
        if not self._model or not self.voice_file:
            return

        try:
            voice_path = Path(self.voice_file)
            if not voice_path.exists():
                print(f"[!] Voice file not found: {self.voice_file}")
                return

            # Load transcript
            prompt_text = ""
            if self.voice_text:
                text_path = Path(self.voice_text)
                if text_path.exists():
                    with open(text_path, "r", encoding="utf-8") as f:
                        prompt_text = f.read().strip()
                else:
                    prompt_text = self.voice_text

            # CosyVoice3 uses instruction format
            # Format: "You are a helpful assistant.<|endofprompt|>{transcript}"
            if prompt_text:
                self._prompt_text = (
                    f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
                )
            else:
                self._prompt_text = "You are a helpful assistant.<|endofprompt|>"

            self._voice_path = str(voice_path)
            print(f"[+] Voice reference registered: {voice_path.name}")

        except Exception as e:
            print(f"[!] Failed to register voice: {e}")
            self._voice_path = None
            self._prompt_text = None

    async def speak(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            WAV audio data as bytes, or None if synthesis fails
        """
        if not self._initialized:
            await self.initialize()

        if not self._available or not self._model:
            return None

        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(None, self._synthesize_sync, text)

        if audio_data:
            await self._audio_queue.put(audio_data)

        return audio_data

    def _synthesize_sync(self, text: str) -> Optional[bytes]:
        """Synchronous speech synthesis."""
        if not self._model:
            return None

        try:
            import torch
            import soundfile as sf

            # Choose synthesis method based on whether we have a reference voice
            if hasattr(self, "_voice_path") and self._voice_path:
                # Zero-shot voice cloning
                audio_generator = self._model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=self._prompt_text,
                    prompt_wav=self._voice_path,
                    stream=False,
                )
            else:
                # Use cross-lingual mode with default voice
                # This uses the model's built-in voice
                audio_generator = self._model.inference_cross_lingual(
                    tts_text=f"You are a helpful assistant.<|endofprompt|>{text}",
                    prompt_wav="",  # Empty for default
                    stream=False,
                )

            # Collect audio chunks
            audio_chunks = []
            for _, result in enumerate(audio_generator):
                if "tts_speech" in result:
                    audio_chunks.append(result["tts_speech"])

            if not audio_chunks:
                print("[!] TTS: No audio generated")
                return None

            # Concatenate all chunks
            audio_tensor = torch.cat(audio_chunks, dim=-1)
            audio_np = audio_tensor.cpu().numpy()

            # Ensure correct shape (should be 1D or 2D with shape [1, samples])
            if audio_np.ndim == 2:
                audio_np = audio_np.squeeze(0)

            # Convert to WAV bytes
            buffer = BytesIO()
            sf.write(buffer, audio_np, self._sample_rate, format="WAV")
            buffer.seek(0)

            return buffer.read()

        except Exception as e:
            print(f"[!] TTS synthesis error: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def get_next_audio(self) -> Optional[bytes]:
        """Get the next audio chunk from the queue."""
        try:
            return self._audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self._sample_rate

    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._available
