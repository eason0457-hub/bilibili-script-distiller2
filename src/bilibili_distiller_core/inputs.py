from __future__ import annotations

import hashlib
import re
import urllib.request

from .models import VideoRef


BV_RE = re.compile(r"\b(BV[0-9A-Za-z]{10})\b")
AV_RE = re.compile(r"\b(?:av|AV)(\d+)\b")
NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:(?:\d+)\s*[.\uFF0E\u3001)\uFF09:]|[\uFF08(]\s*\d+\s*[\uFF09)])\s*"
)


def parse_video_inputs(raw_input: str) -> list[str]:
    """Parse newline, whitespace, comma, and numbered input without duplicates."""
    items: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_input.splitlines():
        line = NUMBER_PREFIX_RE.sub("", raw_line.strip(), count=1).strip()
        for token in re.split(r"[\s,\uFF0C]+", line):
            token = NUMBER_PREFIX_RE.sub("", token.strip(), count=1).strip()
            if not token or re.fullmatch(r"\d+[.\uFF0E\u3001)\uFF09:]?", token):
                continue
            if token not in seen:
                seen.add(token)
                items.append(token)
    return items


def is_supported_input(value: str) -> bool:
    value = value.strip()
    return bool(
        re.match(r"^https?://\S+$", value, re.IGNORECASE)
        or re.fullmatch(r"BV[0-9A-Za-z]{10}", value)
        or re.fullmatch(r"(?:av|AV)\d+", value)
    )


def resolve_video_input(value: str) -> VideoRef:
    value = value.strip()
    if not is_supported_input(value):
        raise ValueError("input must be a Bilibili URL, BV ID, or AV ID")

    bv_match = BV_RE.search(value)
    if bv_match:
        video_id = bv_match.group(1)
        return VideoRef(
            value, f"https://www.bilibili.com/video/{video_id}/", video_id, video_id
        )

    av_match = AV_RE.search(value)
    if av_match:
        video_id = f"av{av_match.group(1)}"
        return VideoRef(
            value, f"https://www.bilibili.com/video/{video_id}/", video_id, video_id
        )

    if re.match(r"^https?://(?:www\.)?b23\.tv/", value, re.IGNORECASE):
        request = urllib.request.Request(value, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            final_url = response.geturl()
        return _from_resolved_url(value, final_url)

    if re.match(r"^https?://(?:www\.)?bilibili\.com/", value, re.IGNORECASE):
        return _from_resolved_url(value, value)

    raise ValueError("URL must use bilibili.com or b23.tv")


def _from_resolved_url(raw_input: str, url: str) -> VideoRef:
    bv_match = BV_RE.search(url)
    av_match = AV_RE.search(url)
    video_id = (
        bv_match.group(1)
        if bv_match
        else (f"av{av_match.group(1)}" if av_match else None)
    )
    output_key = video_id or f"url-{hashlib.sha256(url.encode()).hexdigest()[:12]}"
    return VideoRef(raw_input, url, video_id, output_key)
