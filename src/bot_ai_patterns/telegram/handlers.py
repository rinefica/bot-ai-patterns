import asyncio

from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from bot_ai_patterns.agent import Agent, ContextOverflowError
from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import CONTEXT_LIMIT, CONTEXT_WARN_THRESHOLD
from bot_ai_patterns.context_strategies import (
    BranchingStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)
from bot_ai_patterns.prompts import DEV_PLAN_SYSTEM
from bot_ai_patterns.storage import JSONStorage
from bot_ai_patterns.strategies import run as run_strategy
from bot_ai_patterns.utils import html_to_telegram, sanitize

router = Router()
_client = get_client()
_storage = JSONStorage()

_agents: dict[int, Agent] = {}

_STRATEGY_HELP = (
    "Доступные стратегии:\n"
    "  <code>sliding</code>  — скользящее окно N последних сообщений\n"
    "  <code>facts</code>    — ключевые факты + последние N сообщений\n"
    "  <code>branching</code>— независимые ветки от checkpoint\n\n"
    "Использование: /strategy &lt;название&gt;"
)


class Form(StatesGroup):
    waiting_for_query = State()
    chatting = State()


def _get_agent(user_id: int) -> Agent:
    if user_id not in _agents:
        _agents[user_id] = Agent(
            _client,
            user_id=user_id,
            storage=_storage,
            system_prompt=DEV_PLAN_SYSTEM,
        )
    return _agents[user_id]


def _token_bar(prompt_tokens: int) -> str:
    pct = prompt_tokens / CONTEXT_LIMIT
    filled = int(pct * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {pct:.0%}"


def _format_token_stats(agent: Agent) -> str:
    usage = agent.last_usage
    stats = agent.stats
    if not usage:
        return ""

    warn = ""
    if usage.prompt_tokens >= CONTEXT_WARN_THRESHOLD:
        remaining = CONTEXT_LIMIT - usage.prompt_tokens
        warn = f"\n⚠️ Осталось ~{remaining} токенов до лимита!"

    strategy_lines = "\n".join(agent.strategy.stats_lines())

    return (
        f"\n\n<code>─── токены ───────────────────\n"
        f"{strategy_lines}\n"
        f"Запрос  (контекст): {usage.prompt_tokens:>6}\n"
        f"Ответ   (модель):   {usage.completion_tokens:>6}\n"
        f"Итого   (запрос):   {usage.total_tokens:>6}\n"
        f"─── диалог ({stats.turns} реплик) ──────\n"
        f"История (всего):    {agent.history_len:>6} сообщ.\n"
        f"Потрачено токенов:  {stats.total_tokens:>6}\n"
        f"Стоимость (руб.):   {stats.cost_rub:>6.4f}\n"
        f"Контекст: {_token_bar(usage.prompt_tokens)}</code>"
        f"{warn}"
    )


# ---------------------------------------------------------------------------
# Основные команды
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Привет! Доступные команды:\n"
        "/chat — диалог с агентом\n"
        "/strategy — переключить стратегию контекста\n"
        "/reset — сбросить историю агента\n"
        "/stop — завершить сессию\n\n"
        "Команды ветвления (стратегия branching):\n"
        "/checkpoint — зафиксировать точку ветвления\n"
        "/branch &lt;имя&gt; — создать ветку от checkpoint\n"
        "/switch &lt;имя&gt; — переключиться на ветку\n"
        "/branches — список веток\n\n"
        "Или просто отправь задачу — получишь ответ.",
        parse_mode="HTML",
    )
    await state.set_state(Form.waiting_for_query)


@router.message(Command("chat"))
async def cmd_chat(message: Message, state: FSMContext) -> None:
    await state.set_state(Form.chatting)
    agent = _get_agent(message.from_user.id)
    stats = agent.stats
    turns_info = f" (продолжаем, {stats.turns} реплик)" if stats.turns > 0 else ""
    await message.answer(
        f"Режим диалога активирован{turns_info}.\n"
        f"Текущая стратегия: <b>{agent.strategy.display_name}</b>\n"
        "/strategy — сменить, /reset — сбросить, /stop — выйти.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Управление стратегиями
# ---------------------------------------------------------------------------

@router.message(Command("strategy"))
async def cmd_strategy(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    agent = _get_agent(message.from_user.id)

    if len(parts) == 1:
        await message.answer(
            f"Текущая стратегия: <b>{agent.strategy.display_name}</b>\n\n"
            f"{_STRATEGY_HELP}",
            parse_mode="HTML",
        )
        return

    slug = parts[1].strip().lower()
    if slug in ("sliding", "window", "sliding_window"):
        new_strategy = SlidingWindowStrategy()
    elif slug in ("facts", "sticky", "sticky_facts"):
        new_strategy = StickyFactsStrategy(_client)
    elif slug in ("branching", "branch", "branches"):
        new_strategy = BranchingStrategy()
    else:
        await message.answer(
            f"Неизвестная стратегия: <code>{slug}</code>\n\n{_STRATEGY_HELP}",
            parse_mode="HTML",
        )
        return

    agent.switch_strategy(new_strategy)
    await message.answer(
        f"Стратегия переключена: <b>{new_strategy.display_name}</b>\n"
        "Продолжай диалог — стратегия инициализирована из истории.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Команды ветвления (только для BranchingStrategy)
# ---------------------------------------------------------------------------

def _require_branching(agent: Agent) -> BranchingStrategy | None:
    if isinstance(agent.strategy, BranchingStrategy):
        return agent.strategy
    return None


@router.message(Command("checkpoint"))
async def cmd_checkpoint(message: Message) -> None:
    agent = _get_agent(message.from_user.id)
    bs = _require_branching(agent)
    if bs is None:
        await message.answer(
            "Команда /checkpoint доступна только в стратегии <b>branching</b>.\n"
            "Переключи: /strategy branching",
            parse_mode="HTML",
        )
        return

    bs.set_checkpoint()
    await message.answer(
        f"Checkpoint установлен: {len(bs._checkpoint)} сообщений в ветке [{bs.current_branch_name}].\n"
        "Теперь создай ветку командой /branch &lt;имя&gt;.",
        parse_mode="HTML",
    )


@router.message(Command("branch"))
async def cmd_branch(message: Message) -> None:
    agent = _get_agent(message.from_user.id)
    bs = _require_branching(agent)
    if bs is None:
        await message.answer(
            "Команда /branch доступна только в стратегии <b>branching</b>.",
            parse_mode="HTML",
        )
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer("Укажи имя ветки: /branch &lt;имя&gt;", parse_mode="HTML")
        return

    name = parts[1].strip()
    if not bs.checkpoint_set:
        await message.answer("Сначала установи checkpoint командой /checkpoint.")
        return

    if not bs.create_branch(name):
        await message.answer(f"Ветка <b>{name}</b> уже существует.", parse_mode="HTML")
        return

    await message.answer(
        f"Ветка <b>{name}</b> создана от checkpoint.\n"
        f"Переключись на неё: /switch {name}",
        parse_mode="HTML",
    )


@router.message(Command("switch"))
async def cmd_switch(message: Message) -> None:
    agent = _get_agent(message.from_user.id)
    bs = _require_branching(agent)
    if bs is None:
        await message.answer(
            "Команда /switch доступна только в стратегии <b>branching</b>.",
            parse_mode="HTML",
        )
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer("Укажи имя ветки: /switch &lt;имя&gt;", parse_mode="HTML")
        return

    name = parts[1].strip()
    if not bs.switch_branch(name):
        branches = ", ".join(bs.branch_names)
        await message.answer(
            f"Ветка <b>{name}</b> не найдена.\nДоступные: {branches}",
            parse_mode="HTML",
        )
        return

    await message.answer(
        f"Переключился на ветку <b>{name}</b>.\n"
        f"Сообщений в ветке: {bs.message_count}",
        parse_mode="HTML",
    )


@router.message(Command("branches"))
async def cmd_branches(message: Message) -> None:
    agent = _get_agent(message.from_user.id)
    bs = _require_branching(agent)
    if bs is None:
        await message.answer(
            "Команда /branches доступна только в стратегии <b>branching</b>.",
            parse_mode="HTML",
        )
        return

    lines = []
    for name in bs.branch_names:
        branch = bs._branches[name]
        marker = " ◄ текущая" if name == bs.current_branch_name else ""
        lines.append(f"  <b>{name}</b>: {len(branch.messages)} сообщ.{marker}")

    checkpoint_info = (
        f"\nCheckpoint: {len(bs._checkpoint)} сообщ."
        if bs.checkpoint_set else "\nCheckpoint: не установлен"
    )

    await message.answer(
        f"<b>Ветки диалога:</b>\n" + "\n".join(lines) + checkpoint_info,
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Общие команды
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Обработчики сообщений
# ---------------------------------------------------------------------------

@router.message(Form.chatting)
async def handle_chat(message: Message) -> None:
    user_input = sanitize(message.text or "")
    if not user_input:
        return

    agent = _get_agent(message.from_user.id)
    status = await message.answer("Думаю...")

    try:
        reply, _usage = await asyncio.to_thread(agent.chat, user_input)
    except ContextOverflowError as exc:
        await status.delete()
        await message.answer(
            f"<b>Контекст переполнен!</b>\n\n"
            f"История занимает {exc.prompt_tokens} токенов при лимите {CONTEXT_LIMIT}.\n"
            "Попробуй /strategy sliding (отбросит старые) или /reset.",
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
