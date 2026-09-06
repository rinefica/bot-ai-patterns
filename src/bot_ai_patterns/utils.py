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


def html_to_telegram(html: str) -> str:
    converter = _TelegramHTMLConverter()
    converter.feed(html)
    return converter.result()
