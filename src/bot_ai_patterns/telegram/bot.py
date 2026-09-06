import asyncio
import os

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from bot_ai_patterns.telegram.handlers import router

load_dotenv()


def main() -> None:
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    dp = Dispatcher()
    dp.include_router(router)
    asyncio.run(dp.start_polling(bot))
