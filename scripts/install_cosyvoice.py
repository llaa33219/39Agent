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


def get_python_version() -> tuple[int, int]:
    """Get Python major.minor version."""
    return sys.version_info[:2]


def unpin_version(requirement: str) -> str:
    """Remove exact version pins (==) from a requirement, keep >= constraints."""
    import re

    # Extract package name and any environment markers
    # Pattern: package==version; markers  OR  package==version
    match = re.match(r"^([a-zA-Z0-9_-]+)\s*==\s*[^;]+(;.*)?$", requirement.strip())
    if match:
        pkg_name = match.group(1)
        markers = match.group(2) or ""
        return f"{pkg_name}{markers}\n"

    return requirement


def create_compatible_requirements(original_file: Path, temp_file: Path) -> None:
    """Create a modified requirements file compatible with current Python version."""
    py_version = get_python_version()

    with open(original_file, "r") as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            filtered_lines.append(line)
            continue

        # Skip extra index URLs - they cause conflicts with uv
        if stripped.startswith("--extra-index-url") or stripped.startswith(
            "--index-url"
        ):
            print(f"[*] Skipping index URL (using PyPI): {stripped[:50]}...")
            continue

        # Skip packages that don't work with Python 3.13+
        if py_version >= (3, 13):
            # deepspeed has build issues
            if "deepspeed" in stripped:
                print(f"[*] Skipping deepspeed (build issues on 3.13)")
                continue
            # tensorrt packages don't have 3.13 wheels yet
            if "tensorrt" in stripped:
                print(f"[*] Skipping tensorrt (no 3.13 wheels)")
                continue

        # Remove ALL exact version pins (==) - use latest compatible versions
        if "==" in stripped:
            unpinned = unpin_version(stripped)
            if unpinned != stripped:
                pkg_name = stripped.split("==")[0].split("[")[0].strip()
                print(f"[*] Unpinning: {pkg_name}")
            filtered_lines.append(unpinned)
        else:
            filtered_lines.append(line)

    with open(temp_file, "w") as f:
        f.writelines(filtered_lines)


def install_dependencies():
    """Install CosyVoice Python dependencies."""
    print("\n[3/4] Installing CosyVoice dependencies...")

    requirements_file = REPO_DIR / "requirements.txt"
    if not requirements_file.exists():
        print(f"[!] Requirements file not found: {requirements_file}")
        return

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

    # Create a modified requirements file for compatibility
    temp_requirements = COSYVOICE_DIR / "requirements_compat.txt"
    create_compatible_requirements(requirements_file, temp_requirements)

    # Install from modified requirements.txt
    cmd = []
    if uv_path:
        cmd = [
            uv_path,
            "pip",
            "install",
            "--python",
            sys.executable,
            "-r",
            str(temp_requirements),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(temp_requirements),
            "--quiet",
        ]

    run_command(cmd, check=False)

    # Ensure critical packages are installed
    # These are essential for CosyVoice to work
    critical_packages = [
        ("conformer", "conformer"),
        ("diffusers", "diffusers"),
        ("modelscope", "modelscope"),
        ("hyperpyyaml", "HyperPyYAML"),
        ("whisper", "openai-whisper"),  # Required for voice cloning
        ("inflect", "inflect"),  # Text normalization
        ("librosa", "librosa"),  # Audio processing
        ("torchcodec", "torchcodec"),  # Required by torchaudio for audio loading
    ]

    # Install onnxruntime-gpu on Linux for GPU acceleration
    import platform

    if platform.system() == "Linux":
        try:
            import onnxruntime

            providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise ImportError("No CUDA support")
        except (ImportError, Exception):
            print("[*] Installing onnxruntime-gpu for CUDA support...")
            if uv_path:
                run_command(
                    [
                        uv_path,
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "onnxruntime-gpu",
                    ],
                    check=False,
                )
            else:
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "onnxruntime-gpu",
                        "--quiet",
                    ],
                    check=False,
                )
    else:
        # CPU version for non-Linux
        critical_packages.append(("onnxruntime", "onnxruntime"))

    for import_name, pkg_name in critical_packages:
        try:
            __import__(import_name.lower().replace("-", "_"))
        except ImportError:
            print(f"[*] Installing missing package: {pkg_name}")
            if uv_path:
                run_command(
                    [uv_path, "pip", "install", "--python", sys.executable, pkg_name],
                    check=False,
                )
            else:
                run_command(
                    [sys.executable, "-m", "pip", "install", pkg_name, "--quiet"],
                    check=False,
                )

    # Cleanup temp file
    if temp_requirements.exists():
        temp_requirements.unlink()

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
