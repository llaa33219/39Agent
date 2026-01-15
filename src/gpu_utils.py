"""
GPU detection and device allocation utilities.

Handles multi-GPU setups by assigning different models to different GPUs
to avoid memory conflicts.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class GPUAllocation:
    """GPU device allocation for different models."""

    llm_device: str  # Device for Vision-Language Model
    tts_device: str  # Device for TTS model
    num_gpus: int  # Total number of available GPUs
    has_cuda: bool  # Whether CUDA/ROCm is available


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get memory info for a specific GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        if device_id >= torch.cuda.device_count():
            return {"available": False, "error": f"Device {device_id} not found"}

        props = torch.cuda.get_device_properties(device_id)
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        total = props.total_memory

        return {
            "available": True,
            "device_id": device_id,
            "name": props.name,
            "total_mb": total / (1024 * 1024),
            "allocated_mb": allocated / (1024 * 1024),
            "reserved_mb": reserved / (1024 * 1024),
            "free_mb": (total - reserved) / (1024 * 1024),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def get_gpu_allocation() -> GPUAllocation:
    """
    Determine optimal GPU allocation for LLM and TTS models.

    Strategy:
    - 0 GPUs: Both on CPU
    - 1 GPU: LLM on GPU, TTS on CPU (to avoid memory conflicts)
    - 2+ GPUs: LLM on GPU 0, TTS on GPU 1
    """
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if has_cuda else 0
    except ImportError:
        has_cuda = False
        num_gpus = 0

    if num_gpus == 0:
        # No GPU available - both on CPU
        return GPUAllocation(
            llm_device="cpu", tts_device="cpu", num_gpus=0, has_cuda=False
        )
    elif num_gpus == 1:
        # Single GPU - LLM gets priority, TTS on CPU to avoid conflicts
        return GPUAllocation(
            llm_device="cuda:0", tts_device="cpu", num_gpus=1, has_cuda=True
        )
    else:
        # Multiple GPUs - LLM on GPU 0, TTS on CPU
        #
        # Note: We intentionally keep TTS on CPU even with multiple GPUs because:
        # 1. CosyVoice uses Qwen2ForCausalLM.from_pretrained() which initializes
        #    with meta tensors and assumes cuda:0 as the target device
        # 2. Setting CUDA_VISIBLE_DEVICES after torch import breaks meta tensor
        #    migration (PyTorch error: "Cannot copy out of meta tensor")
        # 3. TTS on CPU is fast enough (~0.3x RTF) and doesn't compete with VLM
        #    for GPU memory, improving overall stability
        # 4. The VLM benefits more from GPU acceleration than TTS
        return GPUAllocation(
            llm_device="cuda:0", tts_device="cpu", num_gpus=num_gpus, has_cuda=True
        )


def print_gpu_info():
    """Print detailed GPU information for debugging."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("[GPU] No CUDA/ROCm available - using CPU")
            return

        num_gpus = torch.cuda.device_count()
        print(f"[GPU] Found {num_gpus} GPU(s)")

        # Check if we're on ROCm
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        backend = "ROCm/HIP" if is_rocm else "CUDA"
        print(f"[GPU] Backend: {backend}")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            print(f"[GPU {i}] {props.name} - {total_gb:.1f} GB")

        allocation = get_gpu_allocation()
        print(
            f"[GPU] Allocation: LLM -> {allocation.llm_device}, TTS -> {allocation.tts_device}"
        )

    except Exception as e:
        print(f"[GPU] Error detecting GPUs: {e}")


def set_device_for_model(device: str):
    """
    Context manager helper to set the default device for model loading.

    Usage:
        with set_device_for_model("cuda:1"):
            model = Model.from_pretrained(...)
    """
    import torch

    class DeviceContext:
        def __init__(self, device: str):
            self.device = device
            self.original_device = None

        def __enter__(self):
            try:
                self.original_device = torch.get_default_device()
            except Exception:
                self.original_device = None

            if self.device != "cpu":
                torch.set_default_device(self.device)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.original_device is not None:
                torch.set_default_device(self.original_device)
            else:
                torch.set_default_device(None)
            return False

    return DeviceContext(device)


# Singleton allocation - computed once at import time
_gpu_allocation: Optional[GPUAllocation] = None


def get_allocation() -> GPUAllocation:
    """Get the cached GPU allocation."""
    global _gpu_allocation
    if _gpu_allocation is None:
        _gpu_allocation = get_gpu_allocation()
    return _gpu_allocation


def get_llm_device() -> str:
    """Get the device string for LLM."""
    return get_allocation().llm_device


def get_tts_device() -> str:
    """Get the device string for TTS."""
    return get_allocation().tts_device
