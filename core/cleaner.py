import re


def clean_text(text: str, max_token_len: int = 100) -> str:
    """Strip base64/hex/token blobs and collapse blank-line runs."""
    if not text:
        return ""
    pattern = re.compile(rf"[A-Za-z0-9+/=_\-]{{{max_token_len},}}")
    cleaned = pattern.sub("[STRIPPED]", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
