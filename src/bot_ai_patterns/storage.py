import json
from pathlib import Path

HISTORY_DIR = Path("data/history")


class JSONStorage:
    """Хранилище состояния агента в JSON-файлах.

    Формат (v3):
    {
        "strategy_name":  "sliding_window",
        "strategy_state": {...},   # состояние стратегии
        "all_messages":   [...]    # аудит-лог всех сообщений (без системного)
    }

    Старые форматы:
    - v1: список сообщений → преобразуется в v3
    - v2: dict с history/summary/recent → преобразуется в v3
    """

    def __init__(self, history_dir: Path = HISTORY_DIR) -> None:
        self._dir = history_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: int) -> Path:
        return self._dir / f"{user_id}.json"

    def load(self, user_id: int) -> dict:
        """Загрузить сохранённое состояние агента.

        Returns:
            dict с ключами strategy_name, strategy_state, all_messages.
            Пустой dict если файла нет.
        """
        path = self._path(user_id)
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))

        # Backward compat: v1 — просто список сообщений
        if isinstance(raw, list):
            return {"strategy_name": None, "strategy_state": None, "all_messages": raw}

        # Backward compat: v2 — dict с history/summary/recent (day9)
        if "history" in raw and "strategy_name" not in raw:
            all_msgs = [m for m in raw.get("history", []) if m.get("role") != "system"]
            return {"strategy_name": None, "strategy_state": None, "all_messages": all_msgs}

        return raw

    def save(
        self,
        user_id: int,
        *,
        strategy_name: str,
        strategy_state: dict,
        all_messages: list[dict[str, str]],
    ) -> None:
        data = {
            "strategy_name": strategy_name,
            "strategy_state": strategy_state,
            "all_messages": all_messages,
        }
        self._path(user_id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete(self, user_id: int) -> None:
        path = self._path(user_id)
        if path.exists():
            path.unlink()
