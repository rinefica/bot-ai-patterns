import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from bot_ai_patterns.artemis.client import get_artemis
from bot_ai_patterns.telegram.handlers import router

load_dotenv()

logger = logging.getLogger(__name__)


async def _run() -> None:
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    dp = Dispatcher()
    dp.include_router(router)

    artemis = get_artemis()
    try:
        await artemis.connect()
        logger.info("Artemis MCP connected")
    except Exception as exc:
        logger.warning("Artemis MCP unavailable: %s", exc)

    try:
        await dp.start_polling(bot)
    finally:
        await artemis.disconnect()


def main() -> None:
    asyncio.run(_run())
