import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from bot_ai_patterns.habits_client import get_habits_client
from bot_ai_patterns.telegram.handlers import router, send_habit_reminder

load_dotenv()

logger = logging.getLogger(__name__)


async def _poll_habits(bot: Bot) -> None:
    """Background task: polls habits-mcp every 30 s and sends pending reminders."""
    client = get_habits_client()
    while True:
        await asyncio.sleep(30)
        if not client.connected:
            continue
        try:
            reminders = await client.get_pending_reminders()
            for reminder in reminders:
                try:
                    await send_habit_reminder(bot, client, reminder)
                except Exception as exc:
                    logger.warning("Failed to deliver reminder %s: %s", reminder.get("id"), exc)
        except Exception as exc:
            logger.warning("Habits poll error: %s", exc)


async def _run() -> None:
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    dp = Dispatcher()
    dp.include_router(router)

    habits = get_habits_client()
    try:
        await habits.connect()
        logger.info("Habits MCP server connected")
    except Exception as exc:
        logger.warning("Habits MCP unavailable: %s", exc)

    poll_task = asyncio.create_task(_poll_habits(bot))
    try:
        await dp.start_polling(bot)
    finally:
        poll_task.cancel()
        await habits.disconnect()


def main() -> None:
    asyncio.run(_run())
