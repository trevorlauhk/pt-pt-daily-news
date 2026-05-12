#!/usr/bin/env python3
"""
PT-PT Daily News Processor – Multi-Article Edition

Fetches the latest news from Notícias ao Minuto RSS (mundo section),
extracts full article text with trafilatura, analyses difficult B2+
vocabulary and conjugated verbs via an LLM (OpenRouter), synthesises
speech with xAI Grok TTS (European Portuguese), and renders an
interactive HTML page with highlighted tooltips and a toggleable English
translation.  Supports browsing 3 articles with a "Next News" button.

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
from pathlib import Path

import feedparser
import requests
import trafilatura
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Data Fetching
# ---------------------------------------------------------------------------


def fetch_articles(count: int = 3) -> list[dict]:
    """
    Iterate over the Notícias ao Minuto 'mundo' RSS feed, skip entries
    whose titles contain video/audio/gallery keywords, and collect up to
    ``count`` articles whose extracted body is longer than 300 characters.
    """
    rss_url = "https://www.noticiasaominuto.com/rss/mundo"
    print(f"[1/4] Fetching RSS feed: {rss_url}")
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        raise RuntimeError("RSS feed contains no entries.")

    skip_keywords = ("vídeo", "video", "áudio", "galeria", "em atualização")
    articles: list[dict] = []

    for entry in feed.entries:
        if len(articles) >= count:
            break

        title = entry.get("title", "").strip()
        if any(kw in title.lower() for kw in skip_keywords):
            continue

        article_url = entry.get("link", "")
        if not article_url:
            continue

        try:
            print(
                f"    Trying article {len(articles) + 1}: "
                f"{title[:60]}{'…' if len(title) > 60 else ''}"
            )
            resp = requests.get(
                article_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"
                },
                timeout=10,
            )
            resp.raise_for_status()
            extracted = trafilatura.extract(resp.text)
            if not extracted:
                continue

            body = extracted.strip()
            if len(body) > 300:
                articles.append(
                    {"title": title, "url": article_url, "body": body}
                )
                print(
                    f"    ✓ Article {len(articles)} collected "
                    f"({len(body)} chars)"
                )
        except Exception as exc:
            print(f"    Warning: failed to fetch {article_url}: {exc}")
            continue

    if not articles:
        raise RuntimeError("No suitable articles found in RSS feed.")

    print(f"    Collected {len(articles)} article(s).")
    return articles


# ---------------------------------------------------------------------------
# 2. LLM Analysis (OpenRouter)
# ---------------------------------------------------------------------------


def analyse_text(title: str, body: str) -> tuple[str, list[dict]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable OPENROUTER_API_KEY is not set."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "PT-PT Daily News",
        },
    )

    system_prompt = (
        "You are a Portuguese-language teaching assistant specialised in "
        "European Portuguese.\n\n"
        "Given a news article (title + body), perform two tasks:\n"
        "1. Produce a high-quality, natural English translation of the "
        "entire article.\n"
        "2. Identify CEFR B2/C1/C2 difficult words and conjugated verbs "
        "from the original Portuguese text.\n\n"
        "Return ONLY a strictly valid JSON object with NO markdown formatting.\n"
        "The JSON object must contain exactly two top-level keys:\n"
        '  "translation": a string containing the full English translation.\n'
        '  "analysis": an array of objects, each with:\n'
        '    "word"       – the word exactly as it appears in the Portuguese text\n'
        '    "infinitive" – the infinitive form if category is "verb", otherwise null\n'
        '    "en"         – concise English translation or explanation\n'
        '    "category"   – either "vocab" or "verb"\n\n'
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

    # Strip accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print("    Raw LLM output:")
        print(raw)
        raise RuntimeError(f"Could not parse LLM JSON output: {exc}")

    if not isinstance(data, dict):
        raise RuntimeError("LLM did not return a JSON object.")

    translation = str(data.get("translation", "")).strip()
    analysis = data.get("analysis", [])
    if not isinstance(analysis, list):
        raise RuntimeError('LLM response "analysis" is not a JSON array.')

    # Normalise analysis items
    normalised: list[dict] = []
    for it in analysis:
        if not isinstance(it, dict):
            continue
        normalised.append(
            {
                "word": str(it.get("word", "")),
                "infinitive": (
                    it.get("infinitive") if it.get("infinitive") else None
                ),
                "en": str(it.get("en", "")),
                "category": (
                    "verb" if it.get("category") == "verb" else "vocab"
                ),
            }
        )

    print(f"    Translation length: {len(translation)} chars")
    print(
        f"    Analysis: {len(normalised)} items "
        f"({sum(1 for i in normalised if i['category'] == 'verb')} verbs, "
        f"{sum(1 for i in normalised if i['category'] == 'vocab')} vocab)."
    )
    return translation, normalised


# ---------------------------------------------------------------------------
# 3. Text-to-Speech (xAI Grok TTS)
# ---------------------------------------------------------------------------


def synthesise_speech(title: str, body: str, out_path: Path) -> None:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable XAI_API_KEY is not set.")

    # Send ONLY the original Portuguese text — never the English translation
    text = f"{title}. {body}"
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
            "voice_id": "ara",
            "language": "pt-PT",
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
    """Escape text then highlight known words with tooltip spans."""
    lookup: dict[str, dict] = {}
    for it in items:
        w = it.get("word", "")
        if w:
            lookup[w.lower()] = it

    if not lookup:
        escaped = html.escape(text)
        return "".join(
            f"<p>{p}</p>" for p in escaped.split("\n\n") if p.strip()
        )

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

    escaped_text = html.escape(text)
    highlighted = pattern.sub(_replace, escaped_text)
    paragraphs = highlighted.split("\n\n")
    return "\n".join(
        f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()
    )


def generate_html_page(articles_data: list[dict], out_path: Path) -> None:
    print("[4/4] Generating HTML page…")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    article_blocks: list[str] = []
    for idx, article in enumerate(articles_data, start=1):
        highlighted = _build_highlighted_html(
            article["body"], article["items"]
        )
        translation_html = "\n".join(
            f"<p>{html.escape(p.strip())}</p>"
            for p in article["translation"].split("\n\n")
            if p.strip()
        )
        display = "block" if idx == 1 else "none"

        block = f"""<article class="card news-article" id="article-{idx}" style="display: {display};">
    <h1>{html.escape(article["title"])}</h1>
    <div class="meta">Notícias ao Minuto · PT-PT Daily News</div>
    <audio controls>
      <source src="{article['audio']}" type="audio/mpeg">
      O seu navegador não suporta o elemento de áudio.
    </audio>
    <button class="btn-translation" data-target="translationBox{idx}">Hide English Translation</button>
    <div id="translationBox{idx}" class="translation-box" style="display: block;">
      {translation_html}
    </div>
    <div class="content">
      {highlighted}
    </div>
  </article>"""
        article_blocks.append(block)

    articles_html = "\n".join(article_blocks)
    total_articles = len(articles_data)

    page = f"""<!DOCTYPE html>
<html lang="pt-PT">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PT-PT Daily News</title>
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
    margin-bottom: 24px;
    border-radius: 10px;
    outline: none;
  }}
  .btn-translation {{
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 0.95rem;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    margin-bottom: 16px;
    font-weight: 500;
  }}
  .btn-translation:hover {{
    background: #1d4ed8;
  }}
  .btn-translation:active {{
    transform: scale(0.98);
  }}
  .translation-box {{
    background: #eff6ff;
    border-left: 4px solid var(--accent);
    padding: 20px 24px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 28px;
    color: #1e3a8a;
    line-height: 1.7;
    font-size: 1rem;
  }}
  .translation-box p {{
    margin: 0 0 0.9em;
    text-align: justify;
  }}
  .translation-box p:last-child {{
    margin-bottom: 0;
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
  #nextBtn {{
    display: block;
    width: 100%;
    margin: 24px 0 0;
    padding: 14px;
    font-size: 1.05rem;
    font-weight: 600;
    color: #fff;
    background: var(--accent);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.2s, opacity 0.2s;
  }}
  #nextBtn:hover {{
    background: #1d4ed8;
  }}
  #nextBtn:disabled {{
    background: #9ca3af;
    cursor: not-allowed;
    opacity: 0.7;
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
    #nextBtn {{ font-size: 0.95rem; padding: 12px; }}
  }}
</style>
</head>
<body>
<div class="container">
  {articles_html}
  <button id="nextBtn">Next News ⏭️</button>
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
<script>
(function() {{
  // Toggle translation for each article
  document.querySelectorAll('.btn-translation').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var targetId = btn.getAttribute('data-target');
      var box = document.getElementById(targetId);
      if (box) {{
        if (box.style.display === 'none') {{
          box.style.display = 'block';
          btn.textContent = 'Hide English Translation';
        }} else {{
          box.style.display = 'none';
          btn.textContent = 'Show English Translation';
        }}
      }}
    }});
  }});

  // Next News navigation
  var current = 1;
  var total = {total_articles};
  var nextBtn = document.getElementById('nextBtn');

  if (nextBtn) {{
    nextBtn.addEventListener('click', function() {{
      if (current >= total) return;

      var currentArticle = document.getElementById('article-' + current);
      if (currentArticle) {{
        currentArticle.style.display = 'none';
      }}

      current++;
      var nextArticle = document.getElementById('article-' + current);
      if (nextArticle) {{
        nextArticle.style.display = 'block';
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}

      if (current >= total) {{
        nextBtn.textContent = 'No more news today';
        nextBtn.disabled = true;
      }}
    }});
  }}
}})();
</script>
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
        articles = fetch_articles(count=3)

        public_html = Path("public_html")
        public_html.mkdir(parents=True, exist_ok=True)

        for i, article in enumerate(articles, start=1):
            print(f"\n--- Processing article {i}/{len(articles)} ---")
            translation, items = analyse_text(article["title"], article["body"])
            article["translation"] = translation
            article["items"] = items

            audio_filename = f"news_{i}.mp3"
            mp3_path = public_html / audio_filename
            synthesise_speech(article["title"], article["body"], mp3_path)
            article["audio"] = audio_filename

        html_path = public_html / "index.html"
        generate_html_page(articles, html_path)

        print("\n✅  Done! Open public_html/index.html in your browser.")
        return 0
    except Exception as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
