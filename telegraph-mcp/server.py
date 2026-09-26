"""telegraph-mcp: создаёт Telegraph-страницы с коллекциями швейных выкроек."""
import base64
import json
import os

import requests
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel
from typing import Annotated
from pydantic import Field

mcp = MCPServer("telegraph-mcp")

_TELEGRAPH_API = "https://api.telegra.ph"
_TELEGRAPH_UPLOAD = "https://telegra.ph/upload"


class UploadResult(BaseModel):
    url: str


class PageResult(BaseModel):
    url: str
    path: str


def _get_token() -> str:
    token = os.environ.get("TELEGRAPH_ACCESS_TOKEN")
    if token:
        return token
    resp = requests.get(
        f"{_TELEGRAPH_API}/createAccount",
        params={"short_name": "SewingBot", "author_name": "Sewing Bot"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegraph createAccount failed: {data.get('error')}")
    return data["result"]["access_token"]


@mcp.tool(
    name="upload_photo",
    description=(
        "Upload a PNG image to Telegraph and return its public URL. "
        "Returns empty url if upload is unavailable."
    ),
)
def upload_photo(
    image_b64: Annotated[str, Field(description="Base64-encoded PNG image bytes.")],
) -> UploadResult:
    image_bytes = base64.b64decode(image_b64)
    try:
        resp = requests.post(
            _TELEGRAPH_UPLOAD,
            files={"file": ("cover.png", image_bytes, "image/png")},
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            if isinstance(data, list) and data:
                return UploadResult(url="https://telegra.ph" + data[0]["src"])
    except Exception:
        pass
    return UploadResult(url="")


@mcp.tool(
    name="create_page",
    description=(
        "Create a Telegraph page with sewing pattern descriptions. "
        "items_json is a JSON array of objects with keys: "
        "title, item_type, summary, tags_json, cover_url, file_name."
    ),
)
def create_page(
    title: Annotated[str, Field(description="Page title (e.g. 'Коллекция выкроек').")],
    items_json: Annotated[
        str,
        Field(
            description=(
                "JSON array of pattern items. Each item: "
                "{title, item_type, summary, tags_json, cover_url, file_name}."
            )
        ),
    ],
) -> PageResult:
    items = json.loads(items_json)
    token = _get_token()

    content: list[dict] = []
    for item in items:
        if item.get("cover_url"):
            content.append({"tag": "figure", "children": [
                {"tag": "img", "attrs": {"src": item["cover_url"]}}
            ]})
        content.append({"tag": "h4", "children": [item.get("title") or "Без названия"]})
        if item.get("item_type"):
            content.append({"tag": "p", "children": [f"Тип: {item['item_type']}"]})
        if item.get("summary"):
            content.append({"tag": "p", "children": [item["summary"]]})
        if item.get("tags_json"):
            try:
                tags = json.loads(item["tags_json"])
                parts = []
                if tags.get("clothing_type"):
                    parts.append(f"👗 {tags['clothing_type']}")
                if tags.get("fabric"):
                    fab = tags["fabric"]
                    parts.append(f"🧵 {', '.join(fab) if isinstance(fab, list) else fab}")
                if tags.get("silhouette"):
                    parts.append(f"📐 {tags['silhouette']}")
                if tags.get("season"):
                    sea = tags["season"]
                    parts.append(f"🌤 {', '.join(sea) if isinstance(sea, list) else sea}")
                if tags.get("style"):
                    parts.append(f"✨ {tags['style']}")
                if parts:
                    content.append({"tag": "p", "children": [" · ".join(parts)]})
            except (json.JSONDecodeError, TypeError):
                pass
        if item.get("file_name"):
            file_url = item.get("file_url", "")
            if file_url:
                content.append({"tag": "p", "children": [
                    "📎 ",
                    {"tag": "a", "attrs": {"href": file_url}, "children": [item["file_name"]]},
                ]})
            else:
                content.append({"tag": "p", "children": [f"📎 {item['file_name']}"]})
        if item.get("attachments"):
            content.append({"tag": "p", "children": ["Прикреплённые файлы:"]})
            for att in item["attachments"]:
                name = att["name"] if isinstance(att, dict) else att
                url = att.get("url", "") if isinstance(att, dict) else ""
                if url:
                    content.append({"tag": "p", "children": [
                        "  📄 ",
                        {"tag": "a", "attrs": {"href": url}, "children": [name]},
                    ]})
                else:
                    content.append({"tag": "p", "children": [f"  📄 {name}"]})
        content.append({"tag": "hr"})

    resp = requests.post(
        f"{_TELEGRAPH_API}/createPage",
        json={
            "access_token": token,
            "title": title[:256],
            "author_name": "Sewing Bot",
            "content": json.dumps(content),
            "return_content": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegraph createPage failed: {result.get('error')}")
    page = result["result"]
    return PageResult(url=page["url"], path=page["path"])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
