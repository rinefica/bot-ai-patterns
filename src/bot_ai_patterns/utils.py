def sanitize(s: str) -> str:
    return s.encode("utf-8", errors="replace").decode("utf-8")
