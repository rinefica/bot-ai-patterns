"""Habit tracker MCP server.

APScheduler (BackgroundScheduler) runs inside the server process and writes
reminder rows to SQLite every minute. The Telegram bot polls
`get_pending_reminders` every 30 s via a long-lived MCP connection.
"""
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Annotated

from apscheduler.schedulers.background import BackgroundScheduler
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

DB_PATH = Path(__file__).parent / "habits.db"
_lock = Lock()
mcp = MCPServer("habits-mcp")
_scheduler = BackgroundScheduler(timezone="UTC")


# ── Database ─────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _lock, _conn() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id           INTEGER PRIMARY KEY,
                habits            TEXT    NOT NULL DEFAULT '[]',
                interval_minutes  INTEGER NOT NULL DEFAULT 2,
                active            INTEGER NOT NULL DEFAULT 1,
                last_reminded_at  TEXT
            );
            CREATE TABLE IF NOT EXISTS check_ins (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     INTEGER NOT NULL,
                checked_at  TEXT    NOT NULL,
                habits      TEXT    NOT NULL,
                completed   TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id  INTEGER NOT NULL,
                due_at   TEXT    NOT NULL,
                sent_at  TEXT
            );
        """)


# ── Scheduler job ─────────────────────────────────────────────────────────────

def _tick() -> None:
    """Runs every minute. Creates reminder rows for users whose interval elapsed."""
    now = datetime.now(UTC)
    with _lock, _conn() as db:
        rows = db.execute(
            "SELECT chat_id, interval_minutes, last_reminded_at FROM users WHERE active=1"
        ).fetchall()
        for row in rows:
            chat_id = row["chat_id"]
            interval = row["interval_minutes"]
            last_at = row["last_reminded_at"]
            if last_at is None:
                due = True
            else:
                due = (now - datetime.fromisoformat(last_at)).total_seconds() >= interval * 60
            if due:
                db.execute(
                    "INSERT INTO reminders (chat_id, due_at) VALUES (?, ?)",
                    (chat_id, now.isoformat()),
                )
                db.execute(
                    "UPDATE users SET last_reminded_at=? WHERE chat_id=?",
                    (now.isoformat(), chat_id),
                )


# ── Result models ─────────────────────────────────────────────────────────────

class TrackingResult(BaseModel):
    chat_id: int
    habits: list[str]
    interval_minutes: int
    status: str


class ReminderItem(BaseModel):
    id: int
    chat_id: int
    due_at: str
    habits: list[str]


class PendingRemindersResult(BaseModel):
    items: list[ReminderItem]


class CheckInResult(BaseModel):
    check_in_id: int
    chat_id: int
    completed: list[str]
    total_habits: int
    completion_rate: float
    checked_at: str


class HabitStat(BaseModel):
    completed: int
    total: int
    rate: float


class SummaryResult(BaseModel):
    chat_id: int
    period: str
    total_check_ins: int
    habit_stats: dict[str, HabitStat]
    overall_rate: float


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="start_tracking",
    description="Register a Telegram chat for periodic habit tracking with reminders.",
)
def start_tracking(
    chat_id: Annotated[int, Field(description="Telegram chat ID.")],
    habits: Annotated[list[str], Field(description="Habits to track (non-empty list).")],
    interval_minutes: Annotated[
        int, Field(description="Reminder interval in minutes.", ge=1, le=1440)
    ] = 2,
) -> TrackingResult:
    if not habits:
        raise ValueError("habits list must not be empty")
    with _lock, _conn() as db:
        db.execute(
            """
            INSERT INTO users (chat_id, habits, interval_minutes, active)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(chat_id) DO UPDATE SET
                habits           = excluded.habits,
                interval_minutes = excluded.interval_minutes,
                active           = 1,
                last_reminded_at = NULL
            """,
            (chat_id, json.dumps(habits, ensure_ascii=False), interval_minutes),
        )
    return TrackingResult(
        chat_id=chat_id, habits=habits,
        interval_minutes=interval_minutes, status="started",
    )


@mcp.tool(
    name="stop_tracking",
    description="Stop habit tracking for a Telegram chat.",
)
def stop_tracking(
    chat_id: Annotated[int, Field(description="Telegram chat ID.")],
) -> TrackingResult:
    with _lock, _conn() as db:
        row = db.execute(
            "SELECT habits, interval_minutes FROM users WHERE chat_id=?", (chat_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"No tracking found for chat_id={chat_id}")
        db.execute("UPDATE users SET active=0 WHERE chat_id=?", (chat_id,))
    return TrackingResult(
        chat_id=chat_id,
        habits=json.loads(row["habits"]),
        interval_minutes=row["interval_minutes"],
        status="stopped",
    )


@mcp.tool(
    name="get_pending_reminders",
    description="Return all reminders that are due but have not been sent yet.",
)
def get_pending_reminders() -> PendingRemindersResult:
    with _lock, _conn() as db:
        rows = db.execute(
            """
            SELECT r.id, r.chat_id, r.due_at, u.habits
            FROM reminders r
            JOIN users u ON u.chat_id = r.chat_id
            WHERE r.sent_at IS NULL
            ORDER BY r.due_at
            """
        ).fetchall()
    return PendingRemindersResult(items=[
        ReminderItem(
            id=r["id"], chat_id=r["chat_id"],
            due_at=r["due_at"], habits=json.loads(r["habits"]),
        )
        for r in rows
    ])


@mcp.tool(
    name="mark_reminder_sent",
    description="Mark a reminder as delivered so it is not returned again.",
)
def mark_reminder_sent(
    reminder_id: Annotated[int, Field(description="ID of the reminder to mark sent.")],
) -> dict:
    sent_at = datetime.now(UTC).isoformat()
    with _lock, _conn() as db:
        db.execute("UPDATE reminders SET sent_at=? WHERE id=?", (sent_at, reminder_id))
    return {"reminder_id": reminder_id, "sent_at": sent_at}


@mcp.tool(
    name="submit_check_in",
    description="Record which habits were completed for this check-in.",
)
def submit_check_in(
    chat_id: Annotated[int, Field(description="Telegram chat ID.")],
    habits: Annotated[list[str], Field(description="All habits that were presented.")],
    completed: Annotated[list[str], Field(description="Habits the user marked as done.")],
) -> CheckInResult:
    now = datetime.now(UTC).isoformat()
    with _lock, _conn() as db:
        cur = db.execute(
            "INSERT INTO check_ins (chat_id, checked_at, habits, completed) VALUES (?, ?, ?, ?)",
            (
                chat_id, now,
                json.dumps(habits, ensure_ascii=False),
                json.dumps(completed, ensure_ascii=False),
            ),
        )
    rate = len(completed) / len(habits) if habits else 0.0
    return CheckInResult(
        check_in_id=cur.lastrowid,
        chat_id=chat_id,
        completed=completed,
        total_habits=len(habits),
        completion_rate=round(rate, 2),
        checked_at=now,
    )


@mcp.tool(
    name="get_summary",
    description="Return aggregated habit completion stats for a given period.",
)
def get_summary(
    chat_id: Annotated[int, Field(description="Telegram chat ID.")],
    period: Annotated[
        str, Field(description="Period: 'today', 'week', or 'all'.")
    ] = "today",
) -> SummaryResult:
    now = datetime.now(UTC)
    if period == "today":
        since = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == "week":
        since = (now - timedelta(days=7)).isoformat()
    else:
        since = "1970-01-01T00:00:00"

    with _lock, _conn() as db:
        rows = db.execute(
            "SELECT habits, completed FROM check_ins WHERE chat_id=? AND checked_at >= ?",
            (chat_id, since),
        ).fetchall()

    counts: dict[str, list] = {}  # habit → [done, total]
    for row in rows:
        for h in json.loads(row["habits"]):
            counts.setdefault(h, [0, 0])
            counts[h][1] += 1
        for h in json.loads(row["completed"]):
            if h in counts:
                counts[h][0] += 1

    habit_stats = {
        h: HabitStat(
            completed=v[0], total=v[1],
            rate=round(v[0] / v[1], 2) if v[1] else 0.0,
        )
        for h, v in counts.items()
    }
    total_done = sum(v[0] for v in counts.values())
    total_all = sum(v[1] for v in counts.values())

    return SummaryResult(
        chat_id=chat_id,
        period=period,
        total_check_ins=len(rows),
        habit_stats=habit_stats,
        overall_rate=round(total_done / total_all, 2) if total_all else 0.0,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _init_db()
    _scheduler.add_job(_tick, "interval", minutes=1, id="habits_tick", replace_existing=True)
    _scheduler.start()
    try:
        mcp.run()
    finally:
        _scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
