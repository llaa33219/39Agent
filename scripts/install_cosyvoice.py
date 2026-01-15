#!/usr/bin/env python3
"""
CosyVoice3 Model Auto-Installer

Installs CosyVoice3 and its dependencies into the data/cosyvoice/ directory.
This script handles:
1. Cloning the CosyVoice GitHub repository (with submodules)
2. Downloading the Fun-CosyVoice3-0.5B-2512 model from HuggingFace
3. Installing Python dependencies

Usage:
    python scripts/install_cosyvoice.py [--force]

Options:
    --force     Force reinstall even if already installed
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
COSYVOICE_DIR = DATA_DIR / "cosyvoice"
REPO_DIR = COSYVOICE_DIR / "CosyVoice"
MODEL_DIR = COSYVOICE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"

# Repository info
COSYVOICE_REPO = "https://github.com/FunAudioLLM/CosyVoice.git"
MODEL_REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def run_command(
    cmd: list[str], cwd: Path = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and print output."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=False,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def is_installed() -> bool:
    """Check if CosyVoice is already installed."""
    # Check for repo
    if not (REPO_DIR / "cosyvoice").exists():
        return False
    # Check for model
    if not (MODEL_DIR / "llm.pt").exists():
        return False
    return True


def clone_repository():
    """Clone the CosyVoice repository with submodules."""
    print("\n[1/4] Cloning CosyVoice repository...")

    if REPO_DIR.exists():
        print(f"[*] Repository already exists at {REPO_DIR}")
        print("[*] Updating submodules...")
        run_command(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_DIR
        )
        return

    # Create parent directory
    COSYVOICE_DIR.mkdir(parents=True, exist_ok=True)

    # Clone with submodules
    run_command(["git", "clone", "--recursive", COSYVOICE_REPO, str(REPO_DIR)])

    print(f"[+] Repository cloned to {REPO_DIR}")


def download_model():
    """Download the Fun-CosyVoice3-0.5B-2512 model from HuggingFace."""
    print("\n[2/4] Downloading model from HuggingFace...")

    if MODEL_DIR.exists() and (MODEL_DIR / "llm.pt").exists():
        print(f"[*] Model already exists at {MODEL_DIR}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] huggingface_hub not installed. Installing...")
        run_command([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import snapshot_download

    # Create model directory
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading {MODEL_REPO_ID}...")
    print("[*] This may take a while (model is ~2GB)...")

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    print(f"[+] Model downloaded to {MODEL_DIR}")


def install_dependencies():
    """Install CosyVoice Python dependencies."""
    print("\n[3/4] Installing CosyVoice dependencies...")

    requirements_file = REPO_DIR / "requirements.txt"
    if not requirements_file.exists():
        print(f"[!] Requirements file not found: {requirements_file}")
        return

    # Read and filter requirements to avoid conflicts
    with open(requirements_file, "r") as f:
        requirements = f.read()

    # Find uv
    uv_path = shutil.which("uv")
    if not uv_path:
        # If uv not in PATH, try to find it relative to user home (fallback)
        uv_path = str(Path.home() / ".local" / "bin" / "uv")
        if not os.path.exists(uv_path):
            uv_path = str(Path.home() / ".cargo" / "bin" / "uv")
            if not os.path.exists(uv_path):
                print("[!] uv not found. Falling back to pip...")
                uv_path = None

    # Install from requirements.txt
    cmd = []
    if uv_path:
        cmd = [
            uv_path,
            "pip",
            "install",
            "--python",
            sys.executable,
            "-r",
            str(requirements_file),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file),
            "--quiet",
        ]

    run_command(cmd, check=False)

    # Ensure critical packages are installed
    critical_packages = [
        "conformer",
        "diffusers",
        "onnxruntime",
        "modelscope",
        "hyperpyyaml",
    ]

    for pkg in critical_packages:
        try:
            __import__(pkg.lower().replace("-", "_"))
        except ImportError:
            print(f"[*] Installing missing package: {pkg}")
            if uv_path:
                run_command(
                    [uv_path, "pip", "install", "--python", sys.executable, pkg],
                    check=False,
                )
            else:
                run_command(
                    [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                    check=False,
                )

    print("[+] Dependencies installed")


def setup_paths():
    """Create necessary path configuration."""
    print("\n[4/4] Setting up paths...")

    # Create a path configuration file that can be imported
    config_content = f'''"""
Auto-generated CosyVoice path configuration.
"""
from pathlib import Path

COSYVOICE_ROOT = Path("{REPO_DIR}")
COSYVOICE_MODEL_DIR = Path("{MODEL_DIR}")
MATCHA_TTS_PATH = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"
'''

    config_file = COSYVOICE_DIR / "paths.py"
    with open(config_file, "w") as f:
        f.write(config_content)

    print(f"[+] Path configuration written to {config_file}")


def verify_installation():
    """Verify that CosyVoice can be imported."""
    print("\n[*] Verifying installation...")

    # Add paths to sys.path temporarily
    sys.path.insert(0, str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR / "third_party" / "Matcha-TTS"))

    try:
        from cosyvoice.cli.cosyvoice import AutoModel

        print("[+] CosyVoice import successful!")
        return True
    except ImportError as e:
        print(f"[!] CosyVoice import failed: {e}")
        print("[!] Some dependencies may be missing. TTS will fall back to CPU mode.")
        return False
    finally:
        # Clean up sys.path
        if str(REPO_DIR) in sys.path:
            sys.path.remove(str(REPO_DIR))
        matcha_path = str(REPO_DIR / "third_party" / "Matcha-TTS")
        if matcha_path in sys.path:
            sys.path.remove(matcha_path)


def main():
    """Main installation routine."""
    force = "--force" in sys.argv

    print("=" * 60)
    print("CosyVoice3 Model Installer")
    print("=" * 60)
    print(f"Installation directory: {COSYVOICE_DIR}")
    print()

    if force and COSYVOICE_DIR.exists():
        print("[*] Force reinstall requested. Cleaning existing installation...")
        # Only remove model, keep repo for faster reinstall
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)

    try:
        clone_repository()
        download_model()
        install_dependencies()
        setup_paths()

        print("\n" + "=" * 60)
        if verify_installation():
            print("[+] CosyVoice3 installation completed successfully!")
        else:
            print("[!] CosyVoice3 installed but verification failed.")
            print("[!] TTS may not work correctly.")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n[!] Installation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
