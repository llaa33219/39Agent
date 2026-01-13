from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
CHARACTER_DIR = ROOT_DIR / "character"

VM_DEFAULT_DISK_GB = 30
VM_DEFAULT_RAM_MB = 12288
VM_DEFAULT_VRAM_MB = 125
VM_WIDTH = 960
VM_HEIGHT = 540
VM_FPS = 30

SERVER_PORT = 8039
CONVERSATION_HISTORY_LIMIT = 5


@dataclass
class VMConfig:
    iso_path: Optional[str] = None
    disk_gb: int = VM_DEFAULT_DISK_GB
    ram_mb: int = VM_DEFAULT_RAM_MB
    vram_mb: int = VM_DEFAULT_VRAM_MB
    width: int = VM_WIDTH
    height: int = VM_HEIGHT


@dataclass
class CharacterConfig:
    name: str = "default"
    display_name: str = "Agent"
    image_path: Optional[str] = None
    llm_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    voice_file: Optional[str] = None
    voice_text: Optional[str] = (
        None  # Transcript of voice_file for NeuTTS voice cloning
    )
    system_prompt: str = ""
    personality: str = ""


@dataclass
class SessionConfig:
    character: CharacterConfig = field(default_factory=CharacterConfig)
    vm: VMConfig = field(default_factory=VMConfig)
    task: str = ""


def load_character(name: str) -> CharacterConfig:
    char_dir = CHARACTER_DIR / name
    config_file = char_dir / "config.yaml"

    if not config_file.exists():
        return CharacterConfig(name=name)

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    image_path = None
    for ext in ["png", "jpg", "jpeg", "gif", "webp"]:
        img = char_dir / f"avatar.{ext}"
        if img.exists():
            image_path = str(img)
            break

    voice_file = data.get("voice_file")
    if voice_file and not Path(voice_file).is_absolute():
        voice_file = str(char_dir / voice_file)

    voice_text = data.get("voice_text")
    if voice_text and not Path(voice_text).is_absolute():
        voice_text = str(char_dir / voice_text)
    elif voice_file and not voice_text:
        voice_text_path = Path(voice_file).with_suffix(".txt")
        if voice_text_path.exists():
            voice_text = str(voice_text_path)

    return CharacterConfig(
        name=name,
        display_name=data.get("display_name", name),
        image_path=image_path,
        llm_model=data.get("llm_model", CharacterConfig.llm_model),
        voice_file=voice_file,
        voice_text=voice_text,
        system_prompt=data.get("system_prompt", ""),
        personality=data.get("personality", ""),
    )


def list_characters() -> list[str]:
    if not CHARACTER_DIR.exists():
        return []
    return [d.name for d in CHARACTER_DIR.iterdir() if d.is_dir()]


def list_isos() -> list[str]:
    iso_dir = DATA_DIR / "iso"
    if not iso_dir.exists():
        return []
    return [f.name for f in iso_dir.glob("*.iso")]
