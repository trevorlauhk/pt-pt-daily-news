#!/usr/bin/env python3
"""
PT-PT Daily News Processor

Fetches the latest news from Público RSS, analyses difficult B2+ vocabulary
and conjugated verbs via an LLM (OpenRouter), synthesises speech with
xAI Grok TTS (European Portuguese), and renders an interactive HTML page
with highlighted tooltips.

Required env vars:
    OPENROUTER_API_KEY
    XAI_API_KEY
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import feedparser
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Data Fetching
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Simple std-lib HTML-to-text converter."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def _strip_html(raw: str) -> str:
    parser = _HTMLStripper()
    parser.feed(raw)
    return parser.get_text()


def _fetch_full_text_fallback(article_url: str) -> str | None:
    """Try to extract a fuller article body from the original web page."""
    try:
        resp = requests.get(
            article_url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
        )
        resp.raise_for_status()
        text = resp.text

        # Strategy A: Open Graph description
        og_match = re.search(
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if og_match:
            return html.unescape(og_match.group(1)).strip()

        # Strategy B: collect <p> paragraphs
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', text, re.IGNORECASE | re.DOTALL)
        if paragraphs:
            return " ".join(_strip_html(p) for p in paragraphs).strip()
    except Exception as exc:
        print(f"    Warning: fallback fetch failed: {exc}")
    return None


def fetch_latest_news() -> tuple[str, str, str]:
    rss_url = "https://feeds.feedburner.com/PublicoRSS"
    print(f"[1/4] Fetching RSS feed: {rss_url}")

    feed = feedparser.parse(rss_url)
    if not feed.entries:
        raise RuntimeError("RSS feed contains no entries.")

    entry = feed.entries[0]
    title = _strip_html(entry.get("title", "Sem título")).strip()

    # Prefer content array, then summary, then description
    raw_body = ""
    if "content" in entry:
        raw_body = entry.content[0].value  # type: ignore[index]
    elif "summary" in entry:
        raw_body = entry.summary
    else:
        raw_body = entry.get("description", "")

    body = _strip_html(raw_body).strip()
    body = re.sub(r"\s+", " ", body)

    # If the RSS snippet is too short, try to retrieve the full article
    if len(body) < 300 and entry.get("link"):
        print("    RSS snippet is short; attempting full-article fallback…")
        fallback = _fetch_full_text_fallback(entry.link)
        if fallback:
            body = re.sub(r"\s+", " ", fallback).strip()

    link = entry.get("link", "")
    print(f"    Title: {title[:80]}{'…' if len(title) > 80 else ''}")
    print(f"    Body length: {len(body)} chars")
    return title, body, link


# ---------------------------------------------------------------------------
# 2. LLM Analysis (OpenRouter)
# ---------------------------------------------------------------------------


def analyse_text(title: str, body: str) -> list[dict]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable OPENROUTER_API_KEY is not set.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "PT-PT Daily News",
        },
    )

    system_prompt = (
        "You are a Portuguese-language teaching assistant specialised in European Portuguese.\n"
        "Given a news article, identify:\n"
        "1. Difficult words at CEFR B2 level or above.\n"
        "2. Conjugated verbs (return their infinitive form).\n\n"
        "Return ONLY a strictly valid JSON array with NO markdown formatting.\n"
        "Each object must contain exactly these keys:\n"
        '  "word"       – the word exactly as it appears in the text\n'
        '  "infinitive" – the infinitive form if category is "verb", otherwise null\n'
        '  "en"         – concise English translation or explanation\n'
        '  "category"   – either "vocab" or "verb"\n\n'
        "Rules:\n"
        "- Do NOT wrap the response in markdown code blocks.\n"
        "- Use double quotes for all strings.\n"
        "- Ensure the output is valid JSON."
    )

    user_prompt = f"Title: {title}\n\nContent:\n{body}\n"

    print("[2/4] Analysing text with LLM (OpenRouter)…")
    response = client.chat.completions.create(
        model="mistralai/mistral-large-2407",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        print("    Raw LLM output:")
        print(raw)
        raise RuntimeError(f"Could not parse LLM JSON output: {exc}")

    if not isinstance(items, list):
        raise RuntimeError("LLM did not return a JSON array.")

    # Normalise keys
    normalised = []
    for it in items:
        if not isinstance(it, dict):
            continue
        normalised.append(
            {
                "word": str(it.get("word", "")),
                "infinitive": it.get("infinitive") if it.get("infinitive") else None,
                "en": str(it.get("en", "")),
                "category": "verb" if it.get("category") == "verb" else "vocab",
            }
        )

    print(f"    Identified {len(normalised)} items ({sum(1 for i in normalised if i['category']=='verb')} verbs, "
          f"{sum(1 for i in normalised if i['category']=='vocab')} vocab).")
    return normalised


# ---------------------------------------------------------------------------
# 3. Text-to-Speech (xAI Grok TTS)
# ---------------------------------------------------------------------------


def synthesise_speech(title: str, body: str, out_path: Path) -> None:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable XAI_API_KEY is not set.")

    text = f"{title}. {body}"
    # xAI TTS supports up to ~15 k characters per request
    if len(text) > 14_000:
        text = text[:14_000]
        print("    Note: text truncated to 14,000 characters for TTS.")

    print("[3/4] Synthesising speech via xAI Grok TTS (voice=ara, lang=pt-PT)…")
    resp = requests.post(
        "https://api.x.ai/v1/tts",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "voice_id": "ara",          # clear and professional — good for news
            "language": "pt-PT",        # European Portuguese
        },
        timeout=120,
    )
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    print(f"    Saved audio: {out_path} ({len(resp.content):,} bytes).")


# ---------------------------------------------------------------------------
# 4. HTML Generation
# ---------------------------------------------------------------------------


def _build_highlighted_html(text: str, items: list[dict]) -> str:
    # Build lookup keyed by lowercase word
    lookup: dict[str, dict] = {}
    for it in items:
        w = it.get("word", "")
        if w:
            lookup[w.lower()] = it

    if not lookup:
        # Nothing to highlight — just escape and paragraphise
        escaped = html.escape(text)
        return "".join(f"<p>{p}</p>" for p in escaped.split("\n\n") if p.strip())

    # Sort by length descending so longer phrases match first
    words = sorted(lookup.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in words) + r")\b",
        re.IGNORECASE,
    )

    def _replace(match: re.Match) -> str:
        original = match.group(0)
        data = lookup[original.lower()]
        category = data.get("category", "vocab")
        en_text = html.escape(data.get("en", ""))
        inf = data.get("infinitive")

        tooltip_lines = [en_text]
        if inf:
            tooltip_lines.append(f"Infinitive: {html.escape(str(inf))}")
        tooltip = "<br>".join(tooltip_lines)
        css_class = "vocab" if category == "vocab" else "verb"

        return (
            f'<span class="tooltip {css_class}">'
            f"{html.escape(original)}"
            f'<span class="tooltiptext">{tooltip}</span>'
            f"</span>"
        )

    # Escape raw text first so we don't accidentally inject HTML,
    # then substitute recognised words with tooltip markup.
    escaped_text = html.escape(text)
    highlighted = pattern.sub(_replace, escaped_text)

    # Turn double-newline into paragraphs
    paragraphs = highlighted.split("\n\n")
    return "\n".join(f"<p>{p.strip()}</p>" for p in paragraphs if p.strip())


def generate_html_page(
    title: str, body: str, items: list[dict], out_path: Path
) -> None:
    print("[4/4] Generating HTML page…")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    highlighted = _build_highlighted_html(body, items)

    page = f"""<!DOCTYPE html>
<html lang="pt-PT">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)} — PT-PT Daily News</title>
<style>
  :root {{
    --bg: #f4f6f8;
    --surface: #ffffff;
    --text: #1a1a1a;
    --muted: #6b7280;
    --accent: #2563eb;
    --vocab-bg: #fef3c7;
    --vocab-fg: #92400e;
    --vocab-border: #f59e0b;
    --verb-bg: #fee2e2;
    --verb-fg: #991b1b;
    --verb-border: #ef4444;
    --radius: 14px;
    --shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.75;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{
    max-width: 820px;
    margin: 48px auto;
    padding: 0 20px;
  }}
  .card {{
    background: var(--surface);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 40px;
    transition: transform 0.2s ease;
  }}
  h1 {{
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 12px;
    line-height: 1.25;
    letter-spacing: -0.02em;
  }}
  .meta {{
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .meta::before {{
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--accent);
    border-radius: 50%;
  }}
  audio {{
    width: 100%;
    margin-bottom: 28px;
    border-radius: 10px;
    outline: none;
  }}
  .content p {{
    margin: 0 0 1.35em;
    text-align: justify;
    font-size: 1.05rem;
  }}
  .tooltip {{
    position: relative;
    cursor: help;
    padding: 1px 4px;
    border-radius: 5px;
    font-weight: 600;
    transition: background 0.15s ease;
  }}
  .tooltip.vocab {{
    background-color: var(--vocab-bg);
    border-bottom: 2.5px solid var(--vocab-border);
    color: var(--vocab-fg);
  }}
  .tooltip.vocab:hover {{
    background-color: #fde68a;
  }}
  .tooltip.verb {{
    background-color: var(--verb-bg);
    border-bottom: 2.5px solid var(--verb-border);
    color: var(--verb-fg);
  }}
  .tooltip.verb:hover {{
    background-color: #fecaca;
  }}
  .tooltip .tooltiptext {{
    visibility: hidden;
    opacity: 0;
    width: 240px;
    background-color: #1f2937;
    color: #fff;
    text-align: center;
    border-radius: 8px;
    padding: 10px 12px;
    position: absolute;
    z-index: 20;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%) translateY(4px);
    font-size: 0.82rem;
    font-weight: 400;
    line-height: 1.4;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    transition: all 0.2s ease;
    pointer-events: none;
  }}
  .tooltip .tooltiptext::after {{
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: #1f2937 transparent transparent transparent;
  }}
  .tooltip:hover .tooltiptext {{
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }}
  .legend {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: var(--muted);
  }}
  .legend .dot {{
    width: 10px;
    height: 10px;
    border-radius: 3px;
    display: inline-block;
  }}
  footer {{
    text-align: center;
    margin: 32px 0;
    color: var(--muted);
    font-size: 0.85rem;
  }}
  @media (max-width: 600px) {{
    .container {{ margin: 24px auto; }}
    .card {{ padding: 24px; }}
    h1 {{ font-size: 1.5rem; }}
  }}
</style>
</head>
<body>
<div class="container">
  <article class="card">
    <h1>{html.escape(title)}</h1>
    <div class="meta">Público · PT-PT Daily News</div>
    <audio controls>
      <source src="news.mp3" type="audio/mpeg">
      O seu navegador não suporta o elemento de áudio.
    </audio>
    <div class="content">
      {highlighted}
    </div>
  </article>
  <footer>
    <span class="legend">
      <span class="dot" style="background:#f59e0b;"></span> Difficult vocab
    </span>
    &nbsp;&middot;&nbsp;
    <span class="legend">
      <span class="dot" style="background:#ef4444;"></span> Conjugated verb
    </span>
    <br><br>
    Generated automatically by PT-PT Daily News
  </footer>
</div>
</body>
</html>
"""

    out_path.write_text(page, encoding="utf-8")
    print(f"    Saved HTML: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        title, body, link = fetch_latest_news()
        items = analyse_text(title, body)

        public_html = Path("public_html")
        public_html.mkdir(parents=True, exist_ok=True)

        mp3_path = public_html / "news.mp3"
        synthesise_speech(title, body, mp3_path)

        html_path = public_html / "index.html"
        generate_html_page(title, body, items, html_path)

        print("\n✅  Done! Open public_html/index.html in your browser.")
        return 0
    except Exception as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
