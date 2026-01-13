import asyncio
import os
import platform
import glob
import warnings
from typing import Optional, Generator
from pathlib import Path
from io import BytesIO

import numpy as np


def _configure_espeak_library():
    if platform.system() != "Darwin":
        return

    search_paths = [
        "/opt/homebrew/Cellar/espeak/*/lib/libespeak.*.dylib",
        "/usr/local/Cellar/espeak/*/lib/libespeak.*.dylib",
    ]

    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            try:
                from phonemizer.backend.espeak.wrapper import EspeakWrapper

                EspeakWrapper.set_library(matches[0])
                return
            except Exception:
                pass


_configure_espeak_library()


class TTSEngine:
    def __init__(
        self, voice_file: Optional[str] = None, voice_text: Optional[str] = None
    ):
        self.voice_file = voice_file
        self.voice_text = voice_text
        self._model = None
        self._initialized = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._sample_rate = 24000
        self._ref_codes = None
        self._ref_text = None

    async def initialize(self):
        if self._initialized:
            return

        print("[*] Loading TTS model: neuphonic/neutts-air-q8-gguf")
        print(
            "[*] This may take a while on first run (downloading from Hugging Face)..."
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)
        self._initialized = True
        print("[+] TTS model loaded successfully")

    def _init_sync(self):
        try:
            from llama_cpp import Llama
            from neucodec import NeuCodec
            from phonemizer.backend import EspeakBackend
            import librosa
            import torch

            self._phonemizer = EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True
            )

            print("[*] Loading backbone (GGUF)...")
            self._backbone = Llama.from_pretrained(
                repo_id="neuphonic/neutts-air-q8-gguf",
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if self._has_gpu() else 0,
                n_ctx=2048,
                mlock=True,
                flash_attn=self._has_gpu(),
            )

            print("[*] Loading codec...")
            self._codec = NeuCodec.from_pretrained("neuphonic/neucodec")
            self._codec = self._codec.to("cuda" if self._has_gpu() else "cpu")
            self._codec.eval()

            self._model = True

            if self.voice_file and Path(self.voice_file).exists():
                self._load_reference()

        except ImportError as e:
            print(f"[!] TTS library not available: {e}")
            print(
                "[!] Voice output disabled. Install with: pip install llama-cpp-python neucodec phonemizer"
            )
            self._model = None
        except Exception as e:
            print(f"[!] Failed to load TTS model: {e}")
            import traceback

            traceback.print_exc()
            self._model = None

    def _has_gpu(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_reference(self):
        import librosa
        import torch

        wav, _ = librosa.load(self.voice_file, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)

        if self._has_gpu():
            wav_tensor = wav_tensor.cuda()

        with torch.no_grad():
            self._ref_codes = (
                self._codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
            )
            if self._has_gpu():
                self._ref_codes = self._ref_codes.cpu()

        if self.voice_text and Path(self.voice_text).exists():
            with open(self.voice_text, "r", encoding="utf-8") as f:
                self._ref_text = f.read().strip()
        else:
            self._ref_text = ""

        print(f"[+] Voice reference loaded: {self.voice_file}")

    def _to_phones(self, text: str) -> str:
        phones = self._phonemizer.phonemize([text])
        phones = phones[0].split()
        return " ".join(phones)

    def _decode(self, codes_str: str) -> Optional[np.ndarray]:
        import re
        import torch

        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes_str)]

        if len(speech_ids) == 0:
            return None

        with torch.no_grad():
            codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :]
            if self._has_gpu():
                codes = codes.cuda()
            recon = self._codec.decode_code(codes).cpu().numpy()

        return recon[0, 0, :]

    async def speak(self, text: str) -> Optional[bytes]:
        if not self._initialized:
            await self.initialize()

        if not self._model:
            return None

        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(None, self._synthesize_sync, text)

        if audio_data:
            await self._audio_queue.put(audio_data)

        return audio_data

    def _synthesize_sync(self, text: str) -> Optional[bytes]:
        if not self._model:
            return None

        try:
            import soundfile as sf

            input_text = self._to_phones(text)

            if self._ref_codes is not None and self._ref_text:
                ref_text_phones = self._to_phones(self._ref_text)
                codes_str = "".join(
                    [f"<|speech_{idx}|>" for idx in self._ref_codes.tolist()]
                )
                prompt = (
                    f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phones} {input_text}"
                    f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
                )
            else:
                prompt = (
                    f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{input_text}"
                    f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>"
                )

            output = self._backbone(
                prompt,
                max_tokens=2048,
                temperature=1.0,
                top_k=50,
                stop=["<|SPEECH_GENERATION_END|>"],
            )
            output_str = output["choices"][0]["text"]

            wav = self._decode(output_str)
            if wav is None:
                print("[!] TTS: No valid speech tokens generated")
                return None

            buffer = BytesIO()
            sf.write(buffer, wav, self._sample_rate, format="wav")
            buffer.seek(0)
            return buffer.read()

        except Exception as e:
            print(f"[!] TTS error: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def get_next_audio(self) -> Optional[bytes]:
        try:
            return self._audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_sample_rate(self) -> int:
        return self._sample_rate
