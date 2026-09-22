"""Клиент для MCP-сервера Artemis (автономная мобильная автоматизация Android).

Artemis запускается как stdio-процесс. Клиент держит одно постоянное соединение
на всё время жизни бота и предоставляет async-методы для каждого MCP-инструмента.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_ARTEMIS_COMMAND = "/Users/kate_yal/artemis/.venv/bin/python"
_ARTEMIS_ARGS = ["-m", "mcp_server"]
_ARTEMIS_CWD = "/Users/kate_yal/artemis"


class ArtemisClient:
    """Обёртка над MCP ClientSession для Artemis.

    Использование:
        client = ArtemisClient()
        await client.connect()
        result = await client.run_task("открой приложение Настройки")
        await client.disconnect()

    Или как async context manager:
        async with ArtemisClient() as client:
            result = await client.run_task(...)
    """

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._exit_stack = None

    async def connect(self) -> None:
        """Запустить Artemis-процесс и установить MCP-соединение."""
        from contextlib import AsyncExitStack

        params = StdioServerParameters(
            command=_ARTEMIS_COMMAND,
            args=_ARTEMIS_ARGS,
            env={**os.environ, "PYTHONPATH": _ARTEMIS_CWD},
            cwd=_ARTEMIS_CWD,
        )
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    async def disconnect(self) -> None:
        """Закрыть соединение и завершить процесс Artemis."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    async def __aenter__(self) -> ArtemisClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    @property
    def connected(self) -> bool:
        return self._session is not None

    # ------------------------------------------------------------------
    # Инструменты
    # ------------------------------------------------------------------

    async def run_task(
        self,
        task_desc: str,
        model: str = "Flash",
        device_serial: str | None = None,
        locked_app_package: str | None = None,
    ) -> dict[str, Any]:
        """Запустить автономную задачу на Android-устройстве.

        Возвращает trace_id для отслеживания через manage_task.
        """
        args: dict[str, Any] = {"task_desc": task_desc, "model": model}
        if device_serial:
            args["device_serial"] = device_serial
        if locked_app_package:
            args["locked_app_package"] = locked_app_package
        return await self._call("mobile_run_task", args)

    async def manage_task(
        self,
        action: str,
        trace_id: str,
        instruction: str | None = None,
        release_loop: bool = False,
    ) -> dict[str, Any]:
        """Управление задачей: status / stop / inject_instruction."""
        args: dict[str, Any] = {"action": action, "trace_id": trace_id}
        if instruction:
            args["instruction"] = instruction
        if release_loop:
            args["release_loop"] = release_loop
        return await self._call("mobile_manage_task", args)

    async def get_device_state(
        self,
        view_type: str = "screenshot",
        device_serial: str | None = None,
    ) -> str:
        """Получить скриншот или иерархию UI устройства."""
        args: dict[str, Any] = {"view_type": view_type}
        if device_serial:
            args["device_serial"] = device_serial
        result = await self._call("mobile_get_device_state", args)
        # инструмент возвращает строку, не dict
        if isinstance(result, str):
            return result
        return result.get("content", str(result))

    # ------------------------------------------------------------------
    # Внутренний вызов
    # ------------------------------------------------------------------

    async def _call(self, tool_name: str, args: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("ArtemisClient не подключён. Вызови connect() сначала.")
        result = await self._session.call_tool(tool_name, args)
        # MCP возвращает CallToolResult; извлекаем содержимое первого блока
        if result.content:
            first = result.content[0]
            if hasattr(first, "text"):
                import json
                try:
                    return json.loads(first.text)
                except (ValueError, TypeError):
                    return first.text
        return {}


# Глобальный синглтон — инициализируется при старте бота
_artemis: ArtemisClient | None = None


def get_artemis() -> ArtemisClient:
    global _artemis
    if _artemis is None:
        _artemis = ArtemisClient()
    return _artemis
