import asyncio

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from bot_ai_patterns.agent import Agent, ContextOverflowError
from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import COMPRESS_AFTER_N, CONTEXT_LIMIT, CONTEXT_WARN_THRESHOLD, RECENT_KEEP
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
    chatting = State()


def _get_agent(user_id: int) -> Agent:
    if user_id not in _agents:
        _agents[user_id] = Agent(_client, user_id=user_id, storage=_storage)
    return _agents[user_id]


def _token_bar(prompt_tokens: int) -> str:
    pct = prompt_tokens / CONTEXT_LIMIT
    filled = int(pct * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {pct:.0%}"


def _format_token_stats(agent: Agent) -> str:
    """Форматирует блок статистики токенов + данные компрессии."""
    usage = agent.last_usage
    stats = agent.stats
    if not usage:
        return ""

    warn = ""
    if usage.prompt_tokens >= CONTEXT_WARN_THRESHOLD:
        remaining = CONTEXT_LIMIT - usage.prompt_tokens
        warn = f"\n⚠️ Осталось ~{remaining} токенов до лимита!"

    # Строка с режимом и данными компрессии
    if agent.compression_enabled:
        cs = agent.compress_stats
        mode_line = f"Режим: СЖАТИЕ ON (каждые {COMPRESS_AFTER_N} сообщ., храним {RECENT_KEEP})\n"
        compress_line = (
            f"В контексте: {agent.recent_len} из {agent.history_len} сообщ.\n"
            f"Сжатий: {cs.compressions_done}, сжато сообщ.: {cs.messages_compressed}\n"
        )
        if cs.tokens_saved > 0:
            compress_line += f"Экономия (посл. сжатие): ~{cs.tokens_saved} токенов\n"
        summary_line = f"Резюме: {'есть' if agent.summary else 'нет'}\n"
    else:
        mode_line = "Режим: полная история\n"
        compress_line = f"В контексте: {agent.history_len} из {agent.history_len} сообщ.\n"
        summary_line = ""

    return (
        f"\n\n<code>─── токены ───────────────────\n"
        f"{mode_line}"
        f"Запрос  (контекст): {usage.prompt_tokens:>6}\n"
        f"Ответ   (модель):   {usage.completion_tokens:>6}\n"
        f"Итого   (запрос):   {usage.total_tokens:>6}\n"
        f"─── диалог ({stats.turns} реплик) ──────\n"
        f"{compress_line}"
        f"{summary_line}"
        f"Потрачено токенов:  {stats.total_tokens:>6}\n"
        f"Стоимость (руб.):   {stats.cost_rub:>6.4f}\n"
        f"Контекст: {_token_bar(usage.prompt_tokens)}</code>"
        f"{warn}"
    )


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Привет! Доступные команды:\n"
        "/chat — диалог с агентом (полная история)\n"
        "/compress — включить/выключить сжатие истории\n"
        "/reset — сбросить историю агента\n"
        "/stop — завершить сессию\n\n"
        "Или просто отправь задачу — получишь 3 варианта ответа."
    )
    await state.set_state(Form.waiting_for_query)


@router.message(Command("chat"))
async def cmd_chat(message: Message, state: FSMContext) -> None:
    await state.set_state(Form.chatting)
    agent = _get_agent(message.from_user.id)
    stats = agent.stats
    turns_info = f" (продолжаем, {stats.turns} реплик в памяти)" if stats.turns > 0 else ""
    mode = "СЖАТИЕ ON" if agent.compression_enabled else "полная история"
    await message.answer(
        f"Режим диалога активирован{turns_info}.\n"
        f"Текущий режим контекста: <b>{mode}</b>\n"
        "После каждого ответа показываю статистику токенов.\n"
        "/compress — переключить сжатие, /reset — сбросить историю, /stop — выйти.",
        parse_mode="HTML",
    )


@router.message(Command("compress"))
async def cmd_compress(message: Message) -> None:
    agent = _get_agent(message.from_user.id)
    enabled = agent.toggle_compression()
    mode = "включено" if enabled else "выключено"
    icon = "✅" if enabled else "❌"

    details = ""
    if enabled:
        details = (
            f"\n\nСжатие срабатывает после каждых {COMPRESS_AFTER_N} сообщений.\n"
            f"В контексте остаётся последние {RECENT_KEEP} сообщения + резюме.\n"
            f"Это уменьшает prompt_tokens и стоимость запроса."
        )
    else:
        details = "\n\nАгент снова отправляет полную историю на каждый запрос."

    await message.answer(f"{icon} Сжатие истории <b>{mode}</b>.{details}", parse_mode="HTML")


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    user_id = message.from_user.id
    if user_id in _agents:
        _agents[user_id].reset()
    await message.answer("История агента сброшена.")


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
    status_text = "Думаю (сжимаю контекст)..." if agent.compression_enabled else "Думаю..."
    status = await message.answer(status_text)

    try:
        reply, _usage = await asyncio.to_thread(agent.chat, user_input)
    except ContextOverflowError as exc:
        await status.delete()
        await message.answer(
            f"<b>Контекст переполнен!</b>\n\n"
            f"История занимает {exc.prompt_tokens} токенов при лимите {CONTEXT_LIMIT}.\n"
            f"Попробуй /compress для сжатия или /reset для сброса.",
            parse_mode="HTML",
        )
        return

    token_info = _format_token_stats(agent)
    await status.delete()
    await message.answer(reply + token_info, parse_mode="HTML")


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
