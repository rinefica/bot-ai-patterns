import html
from html.parser import HTMLParser


def sanitize(s: str) -> str:
    return s.encode("utf-8", errors="replace").decode("utf-8")


class _TelegramHTMLConverter(HTMLParser):
    _KEEP = {"b", "i", "u", "s", "code", "pre", "a"}
    _TO_B = {"strong"}
    _TO_I = {"em"}
    _BLOCK = {"p", "div", "ul", "ol", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._KEEP:
            attr_str = ""
            if tag == "a":
                href = dict(attrs).get("href", "")
                attr_str = f' href="{href}"'
            self._parts.append(f"<{tag}{attr_str}>")
        elif tag in self._TO_B:
            self._parts.append("<b>")
        elif tag in self._TO_I:
            self._parts.append("<i>")
        elif tag == "li":
            self._parts.append("\n• ")
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._KEEP:
            self._parts.append(f"</{tag}>")
        elif tag in self._TO_B:
            self._parts.append("</b>")
        elif tag in self._TO_I:
            self._parts.append("</i>")
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(html.escape(data))

    def result(self) -> str:
        return "".join(self._parts).strip()


_TG_MAX = 4096


def select_relevant(entries: dict[str, str], query: str, top_n: int = 6) -> dict[str, str]:
    """Выбрать наиболее релевантные записи по пересечению токенов с запросом.

    Если совпадений нет — возвращает первые top_n записей (не отбрасывать всё).
    """
    query_tokens = set(query.lower().split())
    scored = [
        (len(query_tokens & set((k + " " + v).lower().split())), k, v)
        for k, v in entries.items()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    with_match = [(k, v) for score, k, v in scored if score > 0]
    result = with_match if with_match else [(k, v) for _, k, v in scored]
    return dict(result[:top_n])


def split_message(text: str, limit: int = _TG_MAX) -> list[str]:
    """Разбить текст на части не длиннее limit символов, по переносам строк."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            parts.append("".join(current))
            current = []
            current_len = 0
        # если одна строка длиннее лимита — режем жёстко
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        current.append(line)
        current_len += len(line)
    if current:
        parts.append("".join(current))
    return parts


def html_to_telegram(html: str) -> str:
    converter = _TelegramHTMLConverter()
    converter.feed(html)
    return converter.result()
