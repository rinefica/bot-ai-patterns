import json
from pathlib import Path

HISTORY_DIR = Path("data/history")


class JSONStorage:
    """Хранилище истории диалогов в JSON-файлах.

    Каждый пользователь — отдельный файл data/history/{user_id}.json.

    Формат файла (v2):
    {
        "history": [...],   # полная история (все сообщения)
        "summary": "...",   # резюме сжатых сообщений (пусто если нет компрессии)
        "recent":  [...]    # последние N сообщений для компрессионного режима
    }

    Поддерживает старый формат v1 (просто список) для обратной совместимости.
    """

    def __init__(self, history_dir: Path = HISTORY_DIR) -> None:
        self._dir = history_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: int) -> Path:
        return self._dir / f"{user_id}.json"

    def load(self, user_id: int) -> dict:
        """Вернуть сохранённые данные.

        Returns:
            dict с ключами 'history', 'summary', 'recent'.
            Пустой dict если файла нет.
        """
        path = self._path(user_id)
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Обратная совместимость: старый формат — просто список сообщений
        if isinstance(raw, list):
            return {"history": raw, "summary": "", "recent": []}
        return raw

    def save(
        self,
        user_id: int,
        history: list[dict[str, str]],
        *,
        summary: str = "",
        recent: list[dict[str, str]] | None = None,
    ) -> None:
        data = {
            "history": history,
            "summary": summary,
            "recent": recent or [],
        }
        self._path(user_id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete(self, user_id: int) -> None:
        path = self._path(user_id)
        if path.exists():
            path.unlink()
