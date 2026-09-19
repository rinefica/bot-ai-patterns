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
        "/reset — сбросить историю (LTM сохраняется)\n"
        "/stop — завершить сессию\n\n"
        "Задача (FSM):\n"
        "/task — состояние задачи\n"
        "/task new &lt;название&gt; — создать задачу\n"
        "/task advance — следующий этап\n"
        "/task pause / resume — пауза / продолжить\n\n"
        "Профиль и память:\n"
        "/profile — просмотр/редактирование профиля\n"
        "/memory — показать все слои памяти\n"
        "/remember &lt;категория&gt; &lt;ключ&gt;: &lt;значение&gt; — добавить в LTM\n"
        "/forget &lt;категория&gt; &lt;ключ&gt; — удалить из LTM\n\n"
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
    await message.answer(
        "История диалога и рабочая память сброшены.\n"
        "Долговременная память (LTM) сохранена — используй /memory для просмотра.",
    )


# ---------------------------------------------------------------------------
# Профиль пользователя
# ---------------------------------------------------------------------------

from bot_ai_patterns.memory.user_profile import PRESETS  # noqa: E402

_PROFILE_HELP = (
    "<b>Управление профилем:</b>\n\n"
    "  /profile — показать профиль\n"
    "  /profile set &lt;поле&gt; &lt;значение&gt; — установить поле\n"
    "  /profile clear — очистить весь профиль\n"
    "  /profile clear &lt;поле&gt; — удалить одно поле\n"
    f"  /profile preset [{' | '.join(PRESETS)}] — применить пресет\n\n"
    "Стандартные поля: <code>имя</code>, <code>стиль</code>, <code>формат</code>, "
    "<code>уровень</code>, <code>стек</code>, <code>ограничения</code>, <code>язык</code>"
)


@router.message(Command("profile"))
async def cmd_profile(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=2)
    agent = _get_agent(message.from_user.id)
    profile = agent.memory.profile

    # /profile — просмотр
    if len(parts) == 1:
        await message.answer(profile.format_telegram(), parse_mode="HTML")
        return

    sub = parts[1].strip().lower()

    # /profile set <поле> <значение>
    if sub == "set":
        if len(parts) < 3 or not parts[2].strip():
            await message.answer(_PROFILE_HELP, parse_mode="HTML")
            return
        rest = parts[2].strip()
        # "поле значение" — первое слово = поле, остальное = значение
        field_parts = rest.split(maxsplit=1)
        if len(field_parts) < 2:
            await message.answer(
                "Формат: /profile set &lt;поле&gt; &lt;значение&gt;", parse_mode="HTML"
            )
            return
        field, value = field_parts[0].lower(), field_parts[1]
        profile.set(field, value)
        await message.answer(
            f"Профиль обновлён:\n<code>{field}</code>: {value}\n\n"
            "Это предпочтение будет применяться к каждому ответу.",
            parse_mode="HTML",
        )
        return

    # /profile preset <имя>
    if sub == "preset":
        preset_name = parts[2].strip().lower() if len(parts) > 2 else ""
        if not preset_name or not profile.apply_preset(preset_name):
            available = ", ".join(f"<code>{p}</code>" for p in PRESETS)
            await message.answer(
                f"Доступные пресеты: {available}", parse_mode="HTML"
            )
            return
        await message.answer(
            f"Пресет <b>{preset_name}</b> применён.\n\n"
            + profile.format_telegram(),
            parse_mode="HTML",
        )
        return

    # /profile clear [поле]
    if sub == "clear":
        if len(parts) > 2:
            field = parts[2].strip().lower()
            if profile.remove(field):
                await message.answer(f"Поле <code>{field}</code> удалено.", parse_mode="HTML")
            else:
                await message.answer(f"Поле <code>{field}</code> не найдено.", parse_mode="HTML")
        else:
            profile.clear()
            await message.answer("Профиль очищен.")
        return

    # Неизвестная подкоманда
    await message.answer(_PROFILE_HELP, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Команды управления памятью
# ---------------------------------------------------------------------------

_REMEMBER_HELP = (
    "Использование: /remember &lt;категория&gt; &lt;ключ&gt;: &lt;значение&gt;\n\n"
    "Категории:\n"
    "  <code>profile</code>   — предпочтения, язык, уровень опыта\n"
    "  <code>decisions</code> — принятые технические решения\n"
    "  <code>knowledge</code> — факты о проекте/предметной области\n\n"
    "Пример: /remember profile язык: Python"
)


@router.message(Command("memory"))
async def cmd_memory(message: Message) -> None:
    agent = _get_agent(message.from_user.id)
    text = agent.memory.format_telegram()
    await message.answer(text, parse_mode="HTML")


@router.message(Command("remember"))
async def cmd_remember(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=2)
    if len(parts) < 3 or ":" not in parts[2]:
        await message.answer(_REMEMBER_HELP, parse_mode="HTML")
        return

    category = parts[1].strip().lower()
    rest = parts[2]
    key, _, value = rest.partition(":")
    key, value = key.strip(), value.strip()

    if not key or not value:
        await message.answer(_REMEMBER_HELP, parse_mode="HTML")
        return

    agent = _get_agent(message.from_user.id)
    try:
        agent.memory.ltm.set(category, key, value)
    except ValueError as exc:
        await message.answer(str(exc), parse_mode="HTML")
        return

    await message.answer(
        f"Записано в LTM [{category}]:\n<code>{key}</code>: {value}",
        parse_mode="HTML",
    )


@router.message(Command("forget"))
async def cmd_forget(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=2)
    if len(parts) < 3:
        await message.answer(
            "Использование: /forget &lt;категория&gt; &lt;ключ&gt;",
            parse_mode="HTML",
        )
        return

    category = parts[1].strip().lower()
    key = parts[2].strip()
    agent = _get_agent(message.from_user.id)

    if agent.memory.ltm.remove(category, key):
        await message.answer(f"Удалено из LTM [{category}]: <code>{key}</code>", parse_mode="HTML")
    else:
        await message.answer(
            f"Ключ <code>{key}</code> не найден в категории [{category}].",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# Команды управления задачей (FSM)
# ---------------------------------------------------------------------------

from bot_ai_patterns.task.task_state import STAGE_ORDER, STAGE_LABELS  # noqa: E402

_TASK_HELP = (
    "<b>Управление задачей:</b>\n\n"
    "  /task — показать состояние\n"
    "  /task new &lt;название&gt; — создать задачу\n"
    "  /task advance — следующий этап\n"
    "  /task step &lt;описание&gt; — установить текущий шаг\n"
    "  /task action &lt;описание&gt; — установить ожидаемое действие\n"
    "  /task note &lt;заметка&gt; — добавить заметку\n"
    "  /task pause — поставить на паузу\n"
    "  /task resume — возобновить (агент продолжит без повторений)\n"
    "  /task clear — удалить задачу\n\n"
    f"Этапы: {' → '.join(STAGE_LABELS[s] for s in STAGE_ORDER)}"
)


@router.message(Command("task"))
async def cmd_task(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=2)
    agent = _get_agent(message.from_user.id)
    tm = agent.memory.task

    # /task — показать состояние
    if len(parts) == 1:
        await message.answer(tm.format_telegram(), parse_mode="HTML")
        return

    sub = parts[1].strip().lower()
    arg = parts[2].strip() if len(parts) > 2 else ""

    # /task new <название>
    if sub == "new":
        if not arg:
            await message.answer("Укажи название: /task new &lt;название&gt;", parse_mode="HTML")
            return
        task = tm.create(arg)
        await message.answer(
            f"Задача создана: <b>{task.title}</b>\n"
            f"Этап: <b>{task.stage_label}</b>\n\n"
            "Теперь диалог с агентом будет вестись в контексте этой задачи.\n"
            "Используй /task advance для перехода на следующий этап.",
            parse_mode="HTML",
        )
        return

    # /task advance
    if sub == "advance":
        if not tm.has_task:
            await message.answer("Нет активной задачи. Создай командой /task new.")
            return
        old_stage = tm.task.stage_label
        if tm.advance():
            await message.answer(
                f"Переход: <b>{old_stage}</b> → <b>{tm.task.stage_label}</b>\n\n"
                + tm.format_telegram(),
                parse_mode="HTML",
            )
        else:
            await message.answer("Задача уже завершена (done).")
        return

    # /task step <описание>
    if sub == "step":
        if not arg:
            await message.answer("Укажи шаг: /task step &lt;описание&gt;", parse_mode="HTML")
            return
        if not tm.has_task:
            await message.answer("Нет активной задачи.")
            return
        tm.set_step(arg)
        await message.answer(f"Текущий шаг: <b>{arg}</b>", parse_mode="HTML")
        return

    # /task action <описание>
    if sub == "action":
        if not arg:
            await message.answer("Укажи действие: /task action &lt;описание&gt;", parse_mode="HTML")
            return
        if not tm.has_task:
            await message.answer("Нет активной задачи.")
            return
        tm.set_action(arg)
        await message.answer(f"Ожидаемое действие: <b>{arg}</b>", parse_mode="HTML")
        return

    # /task note <заметка>
    if sub == "note":
        if not arg:
            await message.answer("Укажи заметку: /task note &lt;текст&gt;", parse_mode="HTML")
            return
        if not tm.has_task:
            await message.answer("Нет активной задачи.")
            return
        tm.add_note(arg)
        await message.answer(f"Заметка добавлена: {arg}")
        return

    # /task pause
    if sub == "pause":
        if tm.pause():
            await message.answer(
                "Задача поставлена на паузу.\n"
                "Возобнови командой /task resume — агент продолжит с текущего шага."
            )
        else:
            await message.answer("Нет активной задачи.")
        return

    # /task resume
    if sub == "resume":
        if tm.resume():
            await message.answer(
                "Задача возобновлена.\n\n"
                + tm.format_telegram()
                + "\n\nОтправь сообщение — агент продолжит без повторений.",
                parse_mode="HTML",
            )
        else:
            await message.answer("Нет активной задачи.")
        return

    # /task clear
    if sub == "clear":
        tm.clear()
        await message.answer("Задача удалена.")
        return

    await message.answer(_TASK_HELP, parse_mode="HTML")


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
