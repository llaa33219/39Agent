#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
VENV_DIR = ROOT_DIR / "data" / "venv"
PYTHON_VERSION = "3.13"

IS_WINDOWS = platform.system() == "Windows"
VENV_BIN_DIR = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")
PYTHON_BIN = VENV_BIN_DIR / ("python.exe" if IS_WINDOWS else "python")


def run_cmd(
    cmd: list[str], check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, **kwargs)


def get_uv_path() -> str | None:
    return shutil.which("uv")


def ensure_uv() -> str:
    uv_path = get_uv_path()
    if uv_path:
        return uv_path

    print("[*] uv not found. Installing...")

    if IS_WINDOWS:
        run_cmd(
            [
                "powershell",
                "-ExecutionPolicy",
                "ByPass",
                "-c",
                "irm https://astral.sh/uv/install.ps1 | iex",
            ]
        )
    else:
        run_cmd(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])

    local_bin = Path.home() / ".local" / "bin"
    if local_bin.exists():
        os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists():
        os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    uv_path = get_uv_path()
    if not uv_path:
        print("[!] Failed to install uv. Please install manually:")
        print("    curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    print("[+] uv installed successfully")
    return uv_path


def ensure_python(uv: str) -> None:
    result = run_cmd(
        [uv, "python", "find", PYTHON_VERSION],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[*] Python {PYTHON_VERSION} not found. Downloading...")
        run_cmd([uv, "python", "install", PYTHON_VERSION])
        print(f"[+] Python {PYTHON_VERSION} installed")
    else:
        print(f"[+] Python {PYTHON_VERSION} found: {result.stdout.strip()}")


def create_venv(uv: str) -> None:
    need_create = False

    if not VENV_DIR.exists():
        need_create = True
    elif not PYTHON_BIN.exists():
        print("[!] Virtual environment is incomplete or corrupted")
        print("[*] Removing and recreating...")
        shutil.rmtree(VENV_DIR)
        need_create = True
    else:
        result = run_cmd(
            [
                str(PYTHON_BIN),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            current_version = result.stdout.strip()
            if not current_version.startswith(PYTHON_VERSION):
                print(f"[!] Venv has Python {current_version}, need {PYTHON_VERSION}")
                print("[*] Recreating virtual environment...")
                shutil.rmtree(VENV_DIR)
                need_create = True

    if need_create:
        print(f"[*] Creating virtual environment with Python {PYTHON_VERSION}...")
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_cmd([uv, "venv", "--python", PYTHON_VERSION, str(VENV_DIR)])
        print(f"[+] Virtual environment created at: {VENV_DIR}")


def detect_gpu() -> str:
    try:
        result = run_cmd(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return "nvidia"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = run_cmd(
            ["rocm-smi"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/bus/pci/devices", "r") as f:
                content = f.read().lower()
                if "amd" in content or "radeon" in content:
                    lspci = run_cmd(
                        ["lspci"], capture_output=True, text=True, check=False
                    )
                    if "AMD" in lspci.stdout or "Radeon" in lspci.stdout:
                        return "amd"
        except:
            pass

    return "cpu"


def install_dependencies(uv: str) -> None:
    requirements_file = ROOT_DIR / "requirements.txt"
    marker_file = VENV_DIR / ".deps_installed"

    if marker_file.exists():
        req_mtime = (
            requirements_file.stat().st_mtime if requirements_file.exists() else 0
        )
        marker_mtime = marker_file.stat().st_mtime
        if marker_mtime >= req_mtime:
            return

    print("[*] Installing dependencies with uv (this is fast!)...")

    gpu_type = detect_gpu()
    print(f"[*] Detected GPU: {gpu_type}")

    torch_index = {
        "nvidia": "https://download.pytorch.org/whl/cu128",
        "amd": "https://download.pytorch.org/whl/rocm6.3",
        "cpu": "https://download.pytorch.org/whl/cpu",
    }[gpu_type]

    print(f"[*] Installing PyTorch with {gpu_type.upper()} support...")
    run_cmd(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(PYTHON_BIN),
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            torch_index,
        ]
    )

    if requirements_file.exists():
        run_cmd(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(PYTHON_BIN),
                "-r",
                str(requirements_file),
            ]
        )

    marker_file.touch()
    print("[+] Dependencies installed successfully")


def check_cosyvoice_dependencies() -> bool:
    critical_packages = ["hyperpyyaml", "onnxruntime", "diffusers", "modelscope"]

    check_script = "; ".join([f"import {pkg}" for pkg in critical_packages])

    try:
        run_cmd([str(PYTHON_BIN), "-c", check_script], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_cosyvoice() -> bool:
    cosyvoice_dir = ROOT_DIR / "data" / "cosyvoice"
    model_dir = cosyvoice_dir / "pretrained_models" / "Fun-CosyVoice3-0.5B"
    install_script = ROOT_DIR / "scripts" / "install_cosyvoice.py"

    # 1. Check if model exists
    model_exists = model_dir.exists() and (model_dir / "llm.pt").exists()

    # 2. Check dependencies
    deps_ok = check_cosyvoice_dependencies()

    if model_exists and deps_ok:
        print("[+] CosyVoice TTS model and dependencies found")
        return True

    if not model_exists:
        print("[*] CosyVoice TTS model not found. Installing...")
    elif not deps_ok:
        print("[*] CosyVoice dependencies missing or broken. Reinstalling...")

    if not install_script.exists():
        print("[!] Install script not found. TTS will not work.")
        return False

    result = run_cmd(
        [str(PYTHON_BIN), str(install_script)],
        check=False,
    )

    if result.returncode != 0:
        print("[!] CosyVoice installation failed. TTS will not work.")
        return False

    print("[+] CosyVoice installed successfully")
    return True


def check_qemu() -> bool:
    try:
        result = run_cmd(
            ["qemu-system-x86_64", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            print(f"[+] QEMU found: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("[!] WARNING: QEMU not found!")
    print("    Please install QEMU:")
    print("    - Ubuntu/Debian: sudo apt install qemu-system-x86 qemu-utils")
    print("    - Fedora: sudo dnf install qemu-system-x86 qemu-img")
    print("    - Arch: sudo pacman -S qemu-full")
    return False


def main():
    print("=" * 50)
    print("  39Agent - AI-Controlled VM Agent")
    print(f"  (Python {PYTHON_VERSION} via uv)")
    print("=" * 50)

    uv = ensure_uv()
    ensure_python(uv)
    create_venv(uv)
    install_dependencies(uv)

    cosyvoice_ok = ensure_cosyvoice()
    if not cosyvoice_ok:
        response = input("\nContinue without TTS? (Voice output will not work) [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)

    qemu_ok = check_qemu()
    if not qemu_ok:
        response = input("\nContinue without QEMU? (VM features will not work) [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)

    print("\n[*] Starting 39Agent server...")
    os.chdir(ROOT_DIR)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    # GPU crash prevention: Configure CUDA/ROCm memory allocator
    gpu_type = detect_gpu()
    if gpu_type == "nvidia":
        # Prevent memory fragmentation and reduce OOM crashes
        # - max_split_size_mb: Limits allocation splitting to reduce fragmentation
        # - garbage_collection_threshold: Triggers cleanup at 80% usage
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:512,garbage_collection_threshold:0.8"
        )
        print("[*] CUDA memory allocator configured for stability")
    elif gpu_type == "amd":
        # ROCm-specific settings for stability
        # - HSA_FORCE_FINE_GRAIN_PCIE: Better memory coherence (reduces HIP errors)
        # - GPU_MAX_HEAP_SIZE: Limit heap to prevent runaway allocations
        env.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")
        env.setdefault("GPU_MAX_HEAP_SIZE", "95")
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:512,garbage_collection_threshold:0.8"
        )
        print("[*] ROCm memory allocator configured for stability")

    run_cmd([str(PYTHON_BIN), "-m", "src.server"], env=env)


if __name__ == "__main__":
    main()
