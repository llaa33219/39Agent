#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
VENV_DIR = ROOT_DIR / "data" / "venv"
PYTHON_BIN = (
    VENV_DIR / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
)
PIP_BIN = VENV_DIR / ("Scripts" if platform.system() == "Windows" else "bin") / "pip"


def detect_gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "nvidia"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/bus/pci/devices", "r") as f:
                content = f.read().lower()
                if "amd" in content or "radeon" in content:
                    lspci = subprocess.run(["lspci"], capture_output=True, text=True)
                    if "AMD" in lspci.stdout or "Radeon" in lspci.stdout:
                        return "amd"
        except:
            pass

    return "cpu"


def create_venv():
    need_create = False

    if not VENV_DIR.exists():
        need_create = True
    elif not PYTHON_BIN.exists():
        print("[!] Virtual environment is incomplete or corrupted")
        print("[*] Removing and recreating...")
        import shutil

        shutil.rmtree(VENV_DIR)
        need_create = True

    if need_create:
        print("[*] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print("[+] Virtual environment created at:", VENV_DIR)


def install_dependencies():
    requirements_file = ROOT_DIR / "requirements.txt"
    marker_file = VENV_DIR / ".deps_installed"

    if marker_file.exists():
        req_mtime = (
            requirements_file.stat().st_mtime if requirements_file.exists() else 0
        )
        marker_mtime = marker_file.stat().st_mtime
        if marker_mtime >= req_mtime:
            return

    print("[*] Installing dependencies...")

    subprocess.run([str(PIP_BIN), "install", "--upgrade", "pip"], check=True)

    gpu_type = detect_gpu()
    print(f"[*] Detected GPU: {gpu_type}")

    if gpu_type == "nvidia":
        print("[*] Installing PyTorch with CUDA support...")
        subprocess.run(
            [
                str(PIP_BIN),
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            check=True,
        )
    elif gpu_type == "amd":
        print("[*] Installing PyTorch with ROCm support...")
        subprocess.run(
            [
                str(PIP_BIN),
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/rocm6.3",
            ],
            check=True,
        )
    else:
        print("[*] Installing PyTorch (CPU only)...")
        subprocess.run(
            [
                str(PIP_BIN),
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            check=True,
        )

    if requirements_file.exists():
        subprocess.run(
            [str(PIP_BIN), "install", "-r", str(requirements_file)], check=True
        )

    if gpu_type == "nvidia":
        print("[*] Installing llama-cpp-python with CUDA support...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        subprocess.run(
            [
                str(PIP_BIN),
                "install",
                "llama-cpp-python",
                "--force-reinstall",
                "--no-cache-dir",
            ],
            env=env,
            check=True,
        )

    marker_file.touch()
    print("[+] Dependencies installed successfully")


def check_espeak():
    try:
        result = subprocess.run(
            ["espeak", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"[+] espeak found")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("[!] WARNING: espeak not found!")
    print("    espeak is required for TTS. Please install:")
    print("    - Ubuntu/Debian: sudo apt install espeak")
    print("    - macOS: brew install espeak")
    print("    - Fedora: sudo dnf install espeak")
    return False


def check_qemu():
    try:
        result = subprocess.run(
            ["qemu-system-x86_64", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
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
    print("=" * 50)

    create_venv()
    install_dependencies()

    espeak_ok = check_espeak()
    if not espeak_ok:
        response = input(
            "\nContinue without espeak? (TTS features will not work) [y/N]: "
        )
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

    subprocess.run([str(PYTHON_BIN), "-m", "src.server"], env=env)


if __name__ == "__main__":
    main()
