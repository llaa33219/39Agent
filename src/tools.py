import asyncio
import re
from dataclasses import dataclass
from typing import Optional, Any, TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from .vm_manager import VMManager
    from .memory import MemoryManager
    from .tts import TTSEngine


@dataclass
class ToolResult:
    success: bool
    message: str
    data: Any = None
    should_continue: bool = True


class ToolExecutor:
    def __init__(
        self,
        vm: "VMManager",
        memory: Optional["MemoryManager"],
        tts: Optional["TTSEngine"],
        on_speak: Optional[Callable[[str], Awaitable[None]]] = None,
        todo_path: str = "todo.md",
    ):
        self.vm = vm
        self.memory = memory
        self.tts = tts
        self.on_speak = on_speak
        self.todo_path = todo_path

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        handlers = {
            "wait": self._tool_wait,
            "input": self._tool_input,
            "text": self._tool_text,
            "cursor-tp": self._tool_cursor_tp,
            "cursor-move": self._tool_cursor_move,
            "click": self._tool_click,
            "click-on": self._tool_click_on,
            "click-off": self._tool_click_off,
            "speak": self._tool_speak,
            "memory": self._tool_memory,
            "todo": self._tool_todo,
            "end": self._tool_end,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(False, f"Unknown tool: {tool_name}")

        try:
            return await handler(params)
        except Exception as e:
            return ToolResult(False, f"Tool error: {str(e)}")

    async def _tool_wait(self, params: dict) -> ToolResult:
        seconds = float(params.get("seconds", 1))
        await asyncio.sleep(seconds)
        return ToolResult(True, f"Waited {seconds} seconds")

    async def _tool_input(self, params: dict) -> ToolResult:
        key_input = params.get("key", "")

        if "+" in key_input:
            keys = [k.strip() for k in key_input.split("+")]
            await self.vm.send_key_combo(keys)
            return ToolResult(True, f"Pressed key combo: {key_input}")
        else:
            await self.vm.send_key(key_input)
            return ToolResult(True, f"Pressed key: {key_input}")

    async def _tool_text(self, params: dict) -> ToolResult:
        text = params.get("text", "")
        await self.vm.type_text(text)
        return ToolResult(True, f"Typed text: {text[:50]}...")

    async def _tool_cursor_tp(self, params: dict) -> ToolResult:
        x = int(params.get("x", 0))
        y = int(params.get("y", 0))
        await self.vm.move_cursor_to(x, y)
        return ToolResult(True, f"Moved cursor to ({x}, {y})")

    async def _tool_cursor_move(self, params: dict) -> ToolResult:
        dx = int(params.get("dx", 0))
        dy = int(params.get("dy", 0))
        await self.vm.move_cursor_relative(dx, dy)
        return ToolResult(True, f"Moved cursor by ({dx}, {dy})")

    async def _tool_click(self, params: dict) -> ToolResult:
        button = params.get("button", "left")
        await self.vm.click(button)
        return ToolResult(True, f"Clicked {button} button")

    async def _tool_click_on(self, params: dict) -> ToolResult:
        button = params.get("button", "left")
        await self.vm.mouse_down(button)
        return ToolResult(True, f"Holding {button} button")

    async def _tool_click_off(self, params: dict) -> ToolResult:
        button = params.get("button", "left")
        await self.vm.mouse_up(button)
        return ToolResult(True, f"Released {button} button")

    async def _tool_speak(self, params: dict) -> ToolResult:
        text = params.get("text", "")

        # 1. Generate Audio First (Blocking/Async wait)
        if self.tts:
            await self.tts.speak(text)

        # 2. Then reflect to UI (Text + Audio queued)
        if self.on_speak:
            await self.on_speak(text)

        return ToolResult(True, f"Spoke: {text[:50]}...")

    async def _tool_memory(self, params: dict) -> ToolResult:
        if not self.memory:
            return ToolResult(False, "Memory system not available")

        query = params.get("query", "")
        results = await self.memory.search(query)

        return ToolResult(True, f"Found {len(results)} memories", data=results)

    async def _tool_todo(self, params: dict) -> ToolResult:
        action = params.get("action", "read")
        content = params.get("content", "")

        if action == "read":
            try:
                with open(self.todo_path, "r", encoding="utf-8") as f:
                    todo_content = f.read()
                return ToolResult(True, "Read todo", data=todo_content)
            except FileNotFoundError:
                return ToolResult(True, "Todo is empty", data="")

        elif action == "write":
            with open(self.todo_path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(True, "Updated todo")

        elif action == "append":
            with open(self.todo_path, "a", encoding="utf-8") as f:
                f.write("\n" + content)
            return ToolResult(True, "Appended to todo")

        return ToolResult(False, f"Unknown todo action: {action}")

    async def _tool_end(self, params: dict) -> ToolResult:
        reason = params.get("reason", "Task completed")
        return ToolResult(True, reason, should_continue=False)


def parse_tool_call(text: str) -> Optional[tuple[str, dict]]:
    """Parse single tool call (deprecated, use parse_tool_calls)."""
    result = parse_tool_calls(text)
    return result[0] if result else None


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Parse all tool calls from response text."""
    import yaml

    pattern = r"<tool>\s*(.*?)\s*</tool>"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return []

    results = []
    for content in matches:
        content = content.strip()
        try:
            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict):
                tool_name = parsed.pop("name", parsed.pop("tool", None))
                if tool_name:
                    results.append((tool_name, parsed))
        except:
            pass

    return results
