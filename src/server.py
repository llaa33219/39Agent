import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .config import (
    ROOT_DIR,
    DATA_DIR,
    CHARACTER_DIR,
    SERVER_PORT,
    SessionConfig,
    VMConfig,
    CharacterConfig,
    load_character,
    list_characters,
    list_isos,
)
from .ai_agent import AIAgent

app = FastAPI(title="39Agent")

WEB_DIR = ROOT_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

active_sessions: dict[str, AIAgent] = {}


class StartSessionRequest(BaseModel):
    character: str
    task: str
    iso_path: Optional[str] = None
    disk_gb: int = 30
    ram_mb: int = 12288
    vram_mb: int = 125


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(WEB_DIR / "templates" / "index.html")


@app.get("/api/characters")
async def get_characters():
    chars = list_characters()
    result = []
    for name in chars:
        char = load_character(name)
        result.append(
            {
                "name": char.name,
                "display_name": char.display_name,
                "has_image": char.image_path is not None,
            }
        )
    return {"characters": result}


@app.get("/api/isos")
async def get_isos():
    return {"isos": list_isos()}


@app.get("/api/character/{name}/image")
async def get_character_image(name: str):
    char = load_character(name)
    if char.image_path:
        return FileResponse(char.image_path)
    return {"error": "No image"}


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    await websocket.accept()

    session_id = None
    agent: Optional[AIAgent] = None

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "start":
                char_config = load_character(data.get("character", "default"))

                iso_name = data.get("iso_path")
                iso_path = None
                if iso_name:
                    iso_full = DATA_DIR / "iso" / iso_name
                    if iso_full.exists():
                        iso_path = str(iso_full)

                vm_config = VMConfig(
                    iso_path=iso_path,
                    disk_gb=data.get("disk_gb", 30),
                    ram_mb=data.get("ram_mb", 12288),
                    vram_mb=data.get("vram_mb", 125),
                )

                session_config = SessionConfig(
                    character=char_config, vm=vm_config, task=data.get("task", "")
                )

                agent = AIAgent(session_config)

                async def on_speak(text: str):
                    await websocket.send_json({"type": "speak", "text": text})

                async def on_tool(name: str, params: dict):
                    await websocket.send_json(
                        {"type": "tool", "name": name, "params": params}
                    )

                async def on_screen(b64_image: str):
                    print(f"[*] Sending screen to browser ({len(b64_image)} bytes)")
                    await websocket.send_json({"type": "screen", "image": b64_image})

                agent.set_callbacks(
                    on_speak=on_speak, on_tool=on_tool, on_screen=on_screen
                )

                await websocket.send_json({"type": "status", "status": "initializing"})

                await agent.initialize()

                await websocket.send_json({"type": "status", "status": "starting_vm"})

                await agent.start_vm()

                session_id = f"session_{id(agent)}"
                active_sessions[session_id] = agent

                await websocket.send_json(
                    {
                        "type": "status",
                        "status": "running",
                        "session_id": session_id,
                        "character": {
                            "name": char_config.name,
                            "display_name": char_config.display_name,
                            "has_image": char_config.image_path is not None,
                        },
                    }
                )

                print("[*] Sending initial screen...")
                initial_screen = await agent.vm.capture_screen()
                import base64
                from io import BytesIO

                buffer = BytesIO()
                initial_screen.save(buffer, format="PNG")
                b64_image = base64.b64encode(buffer.getvalue()).decode()
                await websocket.send_json({"type": "screen", "image": b64_image})
                print(f"[+] Initial screen sent ({len(b64_image)} bytes)")

                asyncio.create_task(run_agent_loop(agent, websocket))

            elif action == "stop":
                if agent:
                    agent.stop()
                    await agent.stop_vm()
                    if session_id and session_id in active_sessions:
                        del active_sessions[session_id]
                    await websocket.send_json({"type": "status", "status": "stopped"})
                    break

            elif action == "restart":
                if agent:
                    agent.stop()
                    await agent.stop_vm()
                    await agent.start_vm()
                    asyncio.create_task(run_agent_loop(agent, websocket))
                    await websocket.send_json({"type": "status", "status": "restarted"})

    except WebSocketDisconnect:
        pass
    finally:
        if agent:
            agent.stop()
            await agent.stop_vm()
        if session_id and session_id in active_sessions:
            del active_sessions[session_id]


async def run_agent_loop(agent: AIAgent, websocket: WebSocket):
    print("[*] run_agent_loop started")
    try:
        async for response, result in agent.run():
            print(f"[*] Agent response received, sending to browser")
            await websocket.send_json(
                {
                    "type": "agent_response",
                    "response": response,
                    "tool_result": {
                        "success": result.success,
                        "message": result.message,
                    }
                    if result
                    else None,
                }
            )
    except Exception as e:
        print(f"[!] Agent loop error: {e}")
        import traceback

        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})


def main():
    import uvicorn

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "templates").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "iso").mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting 39Agent server on http://localhost:{SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)


if __name__ == "__main__":
    main()
