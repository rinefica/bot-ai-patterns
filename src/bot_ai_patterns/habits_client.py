"""Persistent MCP client for habits-mcp server.

A single long-lived subprocess is kept alive for the lifetime of the bot so
that APScheduler inside the server keeps running and fires reminders on time.
"""
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_SERVER_PATH = Path(__file__).parent.parent.parent / "habits-mcp" / "server.py"


class HabitsClient:
    """Manages a long-lived connection to the habits-mcp MCP server."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_PATH)])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def disconnect(self) -> None:
        await self._stack.aclose()
        self._session = None

    async def _call(self, tool: str, args: dict) -> dict:
        if not self._session:
            raise RuntimeError("HabitsClient not connected")
        result = await self._session.call_tool(tool, args)
        if result.is_error:
            raise RuntimeError(f"Tool '{tool}' error: {result.content}")
        for item in result.content:
            raw = getattr(item, "text", None)
            if raw:
                return json.loads(raw)
        return {}

    async def start_tracking(
        self, chat_id: int, habits: list[str], interval_minutes: int = 2
    ) -> dict:
        return await self._call("start_tracking", {
            "chat_id": chat_id,
            "habits": habits,
            "interval_minutes": interval_minutes,
        })

    async def stop_tracking(self, chat_id: int) -> dict:
        return await self._call("stop_tracking", {"chat_id": chat_id})

    async def get_pending_reminders(self) -> list[dict]:
        data = await self._call("get_pending_reminders", {})
        return data.get("items", [])

    async def mark_reminder_sent(self, reminder_id: int) -> None:
        await self._call("mark_reminder_sent", {"reminder_id": reminder_id})

    async def submit_check_in(
        self, chat_id: int, habits: list[str], completed: list[str]
    ) -> dict:
        return await self._call("submit_check_in", {
            "chat_id": chat_id,
            "habits": habits,
            "completed": completed,
        })

    async def get_summary(self, chat_id: int, period: str = "today") -> dict:
        return await self._call("get_summary", {"chat_id": chat_id, "period": period})


_instance: HabitsClient | None = None


def get_habits_client() -> HabitsClient:
    global _instance
    if _instance is None:
        _instance = HabitsClient()
    return _instance
