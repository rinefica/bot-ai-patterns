import asyncio

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from bot_ai_patterns.agent import Agent
from bot_ai_patterns.client import get_client
from bot_ai_patterns.model_comparison import compare_models, format_summary
from bot_ai_patterns.storage import JSONStorage
from bot_ai_patterns.strategies import run as run_strategy
from bot_ai_patterns.utils import html_to_telegram, sanitize

router = Router()
_client = get_client()
_storage = JSONStorage()

# Агент на каждого пользователя (user_id -> Agent)
_agents: dict[int, Agent] = {}


class Form(StatesGroup):
    waiting_for_query = State()
    comparing_models = State()
    chatting = State()


def _get_agent(user_id: int) -> Agent:
    if user_id not in _agents:
        _agents[user_id] = Agent(_client, user_id=user_id, storage=_storage)
    return _agents[user_id]


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Привет! Доступные режимы:\n"
        "/chat — диалог с агентом (с памятью)\n"
        "/reset — сбросить историю агента\n"
        "/compare — сравнить модели разного уровня\n"
        "/stop — завершить сессию\n\n"
        "Или просто отправь задачу — получишь 3 варианта ответа."
    )
    await state.set_state(Form.waiting_for_query)


@router.message(Command("chat"))
async def cmd_chat(message: Message, state: FSMContext) -> None:
    await state.set_state(Form.chatting)
    await message.answer(
        "Режим диалога активирован. Агент помнит контекст разговора.\n"
        "/reset — сбросить историю, /stop — выйти."
    )


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    user_id = message.from_user.id
    if user_id in _agents:
        _agents[user_id].reset()
    await message.answer("История агента сброшена.")


@router.message(Command("compare"))
async def cmd_compare(message: Message, state: FSMContext) -> None:
    await state.set_state(Form.comparing_models)
    await message.answer("Отправь запрос — запущу его на трёх моделях и сравню результаты.")


@router.message(Command("stop"))
async def cmd_stop(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Сессия завершена. Напиши /start чтобы начать снова.")


@router.message(Form.chatting)
async def handle_chat(message: Message) -> None:
    user_input = sanitize(message.text or "")
    if not user_input:
        return

    agent = _get_agent(message.from_user.id)
    status = await message.answer("Думаю...")

    reply = await asyncio.to_thread(agent.chat, user_input)

    await status.delete()
    await message.answer(reply)


@router.message(Form.waiting_for_query)
async def handle_query(message: Message, state: FSMContext) -> None:
    user_message = sanitize(message.text or "")
    if not user_message:
        return

    status = await message.answer("Обрабатываю запрос...")

    result = await run_strategy(_client, user_message)

    await status.delete()

    if result.error:
        await message.answer(f"Ошибка: {result.error}")
    else:
        for label, content in result.sections.items():
            await message.answer(
                f"<b>{label}</b>\n\n{html_to_telegram(content)}",
                parse_mode="HTML",
            )

    await message.answer("Отправь новый запрос или /stop для завершения.")
    await state.set_state(Form.waiting_for_query)


@router.message(Form.comparing_models)
async def handle_compare(message: Message, state: FSMContext) -> None:
    user_message = sanitize(message.text or "")
    if not user_message:
        return

    status = await message.answer("Запрашиваю три модели параллельно...")

    results = await compare_models(_client, user_message)

    await status.delete()

    for r in results:
        if r.error:
            await message.answer(f"<b>{r.label}</b>\n\nОшибка: {r.error}", parse_mode="HTML")
        else:
            await message.answer(
                f"<b>{r.label}</b>\n\n{html_to_telegram(r.response)}",
                parse_mode="HTML",
            )

    summary = format_summary(results)
    await message.answer(
        f"<b>Сравнение</b>\n<pre>{summary}</pre>\n\nОтправь новый запрос или /stop.",
        parse_mode="HTML",
    )
    await state.set_state(Form.comparing_models)
