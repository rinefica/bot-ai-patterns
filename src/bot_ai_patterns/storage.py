import json
from pathlib import Path

HISTORY_DIR = Path("data/history")


class JSONStorage:
    """Хранилище истории диалогов в JSON-файлах.

    Каждый пользователь — отдельный файл data/history/{user_id}.json.
    История загружается при создании агента и сохраняется после каждого сообщения.
    """

    def __init__(self, history_dir: Path = HISTORY_DIR) -> None:
        self._dir = history_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: int) -> Path:
        return self._dir / f"{user_id}.json"

    def load(self, user_id: int) -> list[dict[str, str]]:
        path = self._path(user_id)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, user_id: int, messages: list[dict[str, str]]) -> None:
        self._path(user_id).write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete(self, user_id: int) -> None:
        path = self._path(user_id)
        if path.exists():
            path.unlink()
