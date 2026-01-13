import asyncio
import json
import socket
import subprocess
import tempfile
import struct
from pathlib import Path
from typing import Optional, Any
import io

from PIL import Image

from .config import VMConfig, DATA_DIR


SCANCODE_MAP = {
    "a": 0x1E,
    "b": 0x30,
    "c": 0x2E,
    "d": 0x20,
    "e": 0x12,
    "f": 0x21,
    "g": 0x22,
    "h": 0x23,
    "i": 0x17,
    "j": 0x24,
    "k": 0x25,
    "l": 0x26,
    "m": 0x32,
    "n": 0x31,
    "o": 0x18,
    "p": 0x19,
    "q": 0x10,
    "r": 0x13,
    "s": 0x1F,
    "t": 0x14,
    "u": 0x16,
    "v": 0x2F,
    "w": 0x11,
    "x": 0x2D,
    "y": 0x15,
    "z": 0x2C,
    "1": 0x02,
    "2": 0x03,
    "3": 0x04,
    "4": 0x05,
    "5": 0x06,
    "6": 0x07,
    "7": 0x08,
    "8": 0x09,
    "9": 0x0A,
    "0": 0x0B,
    "space": 0x39,
    "enter": 0x1C,
    "tab": 0x0F,
    "backspace": 0x0E,
    "escape": 0x01,
    "esc": 0x01,
    "f1": 0x3B,
    "f2": 0x3C,
    "f3": 0x3D,
    "f4": 0x3E,
    "f5": 0x3F,
    "f6": 0x40,
    "f7": 0x41,
    "f8": 0x42,
    "f9": 0x43,
    "f10": 0x44,
    "f11": 0x57,
    "f12": 0x58,
    "up": 0x48,
    "down": 0x50,
    "left": 0x4B,
    "right": 0x4D,
    "home": 0x47,
    "end": 0x4F,
    "pageup": 0x49,
    "pagedown": 0x51,
    "insert": 0x52,
    "delete": 0x53,
    "ctrl": 0x1D,
    "alt": 0x38,
    "shift": 0x2A,
    "-": 0x0C,
    "=": 0x0D,
    "[": 0x1A,
    "]": 0x1B,
    "\\": 0x2B,
    ";": 0x27,
    "'": 0x28,
    "`": 0x29,
    ",": 0x33,
    ".": 0x34,
    "/": 0x35,
}

SHIFT_CHARS = {
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "-",
    "+": "=",
    "{": "[",
    "}": "]",
    "|": "\\",
    ":": ";",
    '"': "'",
    "~": "`",
    "<": ",",
    ">": ".",
    "?": "/",
    "A": "a",
    "B": "b",
    "C": "c",
    "D": "d",
    "E": "e",
    "F": "f",
    "G": "g",
    "H": "h",
    "I": "i",
    "J": "j",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "O": "o",
    "P": "p",
    "Q": "q",
    "R": "r",
    "S": "s",
    "T": "t",
    "U": "u",
    "V": "v",
    "W": "w",
    "X": "x",
    "Y": "y",
    "Z": "z",
}


class QMPClient:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self):
        self._reader, self._writer = await asyncio.open_unix_connection(
            self.socket_path
        )
        await self._reader.readline()
        self._writer.write(b'{"execute": "qmp_capabilities"}\n')
        await self._writer.drain()
        await self._reader.readline()

    async def execute(self, command: str, arguments: Optional[dict] = None) -> dict:
        if not self._writer or not self._reader:
            raise RuntimeError("Not connected")
        msg: dict[str, Any] = {"execute": command}
        if arguments:
            msg["arguments"] = arguments
        self._writer.write((json.dumps(msg) + "\n").encode())
        await self._writer.drain()
        response = await self._reader.readline()
        return json.loads(response)

    async def send_key(self, scancode: int, down: bool):
        await self.execute(
            "input-send-event",
            {
                "events": [
                    {
                        "type": "key",
                        "data": {
                            "down": down,
                            "key": {"type": "number", "data": scancode},
                        },
                    }
                ]
            },
        )

    async def send_mouse_move(self, x: int, y: int, absolute: bool = True):
        if absolute:
            await self.execute(
                "input-send-event",
                {
                    "events": [
                        {"type": "abs", "data": {"axis": "x", "value": x}},
                        {"type": "abs", "data": {"axis": "y", "value": y}},
                    ]
                },
            )
        else:
            await self.execute(
                "input-send-event",
                {
                    "events": [
                        {"type": "rel", "data": {"axis": "x", "value": x}},
                        {"type": "rel", "data": {"axis": "y", "value": y}},
                    ]
                },
            )

    async def send_mouse_button(self, button: str, down: bool):
        btn_map = {"left": 0, "right": 1, "middle": 2}
        await self.execute(
            "input-send-event",
            {
                "events": [
                    {"type": "btn", "data": {"down": down, "button": f"mouse-{button}"}}
                ]
            },
        )

    async def close(self):
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()


class VNCClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5900):
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self.width = 0
        self.height = 0

    async def connect(self):
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)

        version = await self._reader.readexactly(12)
        self._writer.write(b"RFB 003.008\n")
        await self._writer.drain()

        num_types = (await self._reader.readexactly(1))[0]
        security_types = await self._reader.readexactly(num_types)
        self._writer.write(bytes([1]))
        await self._writer.drain()

        result = await self._reader.readexactly(4)
        if struct.unpack(">I", result)[0] != 0:
            raise ConnectionError("VNC authentication failed")

        self._writer.write(bytes([1]))
        await self._writer.drain()

        server_init = await self._reader.readexactly(24)
        self.width, self.height = struct.unpack(">HH", server_init[0:4])

        name_len = struct.unpack(">I", server_init[20:24])[0]
        await self._reader.readexactly(name_len)

    async def capture_screen(self) -> Image.Image:
        if not self._writer or not self._reader:
            raise RuntimeError("Not connected")

        msg = struct.pack(">BBHHHH", 3, 0, 0, 0, self.width, self.height)
        self._writer.write(msg)
        await self._writer.drain()

        header = await self._reader.readexactly(4)
        if header[0] != 0:
            raise ValueError(f"Unexpected message type: {header[0]}")

        num_rects = struct.unpack(">H", header[2:4])[0]

        image = Image.new("RGB", (self.width, self.height))

        for _ in range(num_rects):
            rect_header = await self._reader.readexactly(12)
            x, y, w, h, encoding = struct.unpack(">HHHHi", rect_header)

            if encoding == 0:
                pixels = await self._reader.readexactly(w * h * 4)
                rect_img = Image.frombytes("RGBX", (w, h), pixels)
                image.paste(rect_img.convert("RGB"), (x, y))

        return image

    async def close(self):
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()


class VMManager:
    def __init__(self, config: VMConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.qmp: Optional[QMPClient] = None
        self.vnc: Optional[VNCClient] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._qmp_socket: Optional[str] = None
        self._vnc_port = 5900

    async def start(self):
        print("[*] Starting VM...")
        self._temp_dir = tempfile.TemporaryDirectory(prefix="39agent_")
        temp_path = Path(self._temp_dir.name)

        print("[*] Creating virtual disk...")
        disk_path = temp_path / "disk.qcow2"
        subprocess.run(
            [
                "qemu-img",
                "create",
                "-f",
                "qcow2",
                str(disk_path),
                f"{self.config.disk_gb}G",
            ],
            check=True,
            capture_output=True,
        )

        self._qmp_socket = str(temp_path / "qmp.sock")

        # Find OVMF files (different paths on different distros)
        ovmf_paths = [
            # Fedora/RHEL
            (
                "/usr/share/edk2/x64/OVMF_CODE.4m.fd",
                "/usr/share/edk2/x64/OVMF_VARS.4m.fd",
            ),
            # Ubuntu/Debian
            ("/usr/share/OVMF/OVMF_CODE.fd", "/usr/share/OVMF/OVMF_VARS.fd"),
            ("/usr/share/OVMF/OVMF_CODE_4M.fd", "/usr/share/OVMF/OVMF_VARS_4M.fd"),
            # Arch Linux
            (
                "/usr/share/edk2-ovmf/x64/OVMF_CODE.fd",
                "/usr/share/edk2-ovmf/x64/OVMF_VARS.fd",
            ),
        ]

        ovmf_code = None
        ovmf_vars_src = None
        for code_path, vars_path in ovmf_paths:
            if Path(code_path).exists() and Path(vars_path).exists():
                ovmf_code = Path(code_path)
                ovmf_vars_src = Path(vars_path)
                break

        if not ovmf_code or not ovmf_vars_src:
            raise RuntimeError(
                "OVMF firmware not found. Install with:\n"
                "  Ubuntu/Debian: sudo apt install ovmf\n"
                "  Fedora: sudo dnf install edk2-ovmf\n"
                "  Arch: sudo pacman -S edk2-ovmf"
            )

        ovmf_vars = temp_path / ovmf_vars_src.name

        import shutil

        shutil.copy(ovmf_vars_src, ovmf_vars)

        cmd = [
            "qemu-system-x86_64",
            "-enable-kvm",
            "-m",
            str(self.config.ram_mb),
            "-smp",
            "4",
            "-cpu",
            "host",
            "-drive",
            f"if=pflash,format=raw,readonly=on,file={ovmf_code}",
            "-drive",
            f"if=pflash,format=raw,file={ovmf_vars}",
            "-drive",
            f"file={disk_path},format=qcow2,if=virtio",
            "-device",
            f"VGA,vgamem_mb={self.config.vram_mb}",
            "-vnc",
            f"127.0.0.1:{self._vnc_port - 5900}",
            "-qmp",
            f"unix:{self._qmp_socket},server,nowait",
            "-display",
            "none",
            "-usb",
            "-device",
            "usb-tablet",
        ]

        if self.config.iso_path:
            cmd.extend(["-cdrom", self.config.iso_path, "-boot", "d"])

        print("[*] Launching QEMU...")
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        print("[*] Waiting for QEMU to start...")
        await self._wait_for_socket(self._qmp_socket, timeout=10.0)

        print("[*] Connecting to QMP...")
        self.qmp = QMPClient(self._qmp_socket)
        try:
            await self.qmp.connect()
        except (ConnectionRefusedError, FileNotFoundError) as e:
            _, stderr = self.process.communicate(timeout=1)
            if stderr:
                raise RuntimeError(f"QEMU failed to start: {stderr.decode()}") from e
            raise

        print("[*] Connecting to VNC...")
        self.vnc = VNCClient(port=self._vnc_port)
        await self.vnc.connect()
        print(f"[+] VM started successfully (VNC: {self.vnc.width}x{self.vnc.height})")

    async def _wait_for_socket(self, socket_path: str, timeout: float = 10.0):
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            if Path(socket_path).exists():
                return
            if self.process and self.process.poll() is not None:
                _, stderr = self.process.communicate()
                raise RuntimeError(f"QEMU exited early: {stderr.decode()}")
            await asyncio.sleep(0.1)
        raise TimeoutError(f"QEMU socket not ready after {timeout}s")

    async def stop(self):
        if self.vnc:
            await self.vnc.close()
        if self.qmp:
            await self.qmp.close()
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
        if self._temp_dir:
            self._temp_dir.cleanup()

    async def capture_screen(self) -> Image.Image:
        if not self.vnc:
            raise RuntimeError("VM not started")
        return await self.vnc.capture_screen()

    async def send_key(self, key: str, down: Optional[bool] = None):
        if not self.qmp:
            raise RuntimeError("VM not started")

        key_lower = key.lower()
        scancode = SCANCODE_MAP.get(key_lower)

        if scancode is None:
            raise ValueError(f"Unknown key: {key}")

        if down is None:
            await self.qmp.send_key(scancode, True)
            await asyncio.sleep(0.05)
            await self.qmp.send_key(scancode, False)
        else:
            await self.qmp.send_key(scancode, down)

    async def send_key_combo(self, keys: list[str]):
        if not self.qmp:
            raise RuntimeError("VM not started")

        for key in keys:
            scancode = SCANCODE_MAP.get(key.lower())
            if scancode:
                await self.qmp.send_key(scancode, True)
                await asyncio.sleep(0.02)

        for key in reversed(keys):
            scancode = SCANCODE_MAP.get(key.lower())
            if scancode:
                await self.qmp.send_key(scancode, False)
                await asyncio.sleep(0.02)

    async def type_text(self, text: str):
        if not self.qmp:
            raise RuntimeError("VM not started")

        for char in text:
            needs_shift = char in SHIFT_CHARS
            actual_char = SHIFT_CHARS.get(char, char)

            if actual_char == " ":
                actual_char = "space"

            scancode = SCANCODE_MAP.get(actual_char.lower())
            if scancode is None:
                continue

            if needs_shift:
                await self.qmp.send_key(SCANCODE_MAP["shift"], True)

            await self.qmp.send_key(scancode, True)
            await asyncio.sleep(0.02)
            await self.qmp.send_key(scancode, False)

            if needs_shift:
                await self.qmp.send_key(SCANCODE_MAP["shift"], False)

            await asyncio.sleep(0.03)

    async def move_cursor_to(self, x: int, y: int):
        if not self.qmp:
            raise RuntimeError("VM not started")

        abs_x = int((x / self.config.width) * 32767)
        abs_y = int((y / self.config.height) * 32767)
        await self.qmp.send_mouse_move(abs_x, abs_y, absolute=True)

    async def move_cursor_relative(self, dx: int, dy: int):
        if not self.qmp:
            raise RuntimeError("VM not started")
        await self.qmp.send_mouse_move(dx, dy, absolute=False)

    async def click(self, button: str = "left"):
        if not self.qmp:
            raise RuntimeError("VM not started")
        await self.qmp.send_mouse_button(button, True)
        await asyncio.sleep(0.05)
        await self.qmp.send_mouse_button(button, False)

    async def mouse_down(self, button: str = "left"):
        if not self.qmp:
            raise RuntimeError("VM not started")
        await self.qmp.send_mouse_button(button, True)

    async def mouse_up(self, button: str = "left"):
        if not self.qmp:
            raise RuntimeError("VM not started")
        await self.qmp.send_mouse_button(button, False)
