#!/usr/bin/env python3
"""
PT-PT Daily News – Ultimate Edition

Fetches the latest news from Notícias ao Minuto RSS (mundo section),
extracts full article text with trafilatura, downloads representative
images, analyses vocabulary with an LLM (OpenRouter) returning bilingual
translations + sentence alignment + Cantonese teacher summary,
synthesises speech with xAI Grok TTS, renders an interactive HTML page
with dual-layer highlighting & sentence-sync hover, and sends a morning
email digest.

Required env vars:
    OPENROUTER_API_KEY
    XAI_API_KEY
    SMTP_SERVER          (optional, for email)
    EMAIL_USER           (optional, for email)
    EMAIL_PASS           (optional, for email)
    GITHUB_PAGES_URL     (optional, default: https://yourname.github.io/repo)
"""

from __future__ import annotations

import html
import json
import os
import re
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import feedparser
import requests
import trafilatura
from bs4 import BeautifulSoup
from openai import OpenAI

GITHUB_PAGES_URL = os.environ.get(
    "GITHUB_PAGES_URL", "https://yourname.github.io/pt-pt-daily-news"
)

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
    print(f"[1/5] Fetching RSS feed: {rss_url}")
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
                    {
                        "title": title,
                        "url": article_url,
                        "body": body,
                        "raw_html": resp.text,
                    }
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


def fetch_article_image(article: dict, idx: int, out_dir: Path) -> str:
    """
    Try to extract a representative image from the article HTML and save it.
    Returns the filename if successful, otherwise an empty string.
    """
    article_url = article["url"]
    raw_html = article.get("raw_html", "")
    if not raw_html:
        return ""

    filename = f"news_img_{idx}.jpg"
    out_path = out_dir / filename

    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        img_url = None

        # Strategy 1: Open Graph image
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            img_url = og_img["content"]

        # Strategy 2: first substantial <img> inside <article> or content area
        if not img_url:
            container = soup.find("article") or soup.find(
                attrs={"class": re.compile(r"content|article-body|main-content")}
            )
            if container:
                for img in container.find_all("img"):
                    src = img.get("src", "")
                    if src and not any(
                        x in src.lower()
                        for x in ("icon", "logo", "avatar", "share", "social")
                    ):
                        img_url = src
                        break

        # Strategy 3: any <img> with width > 200
        if not img_url:
            for img in soup.find_all("img"):
                w = img.get("width", "")
                src = img.get("src", "")
                if src and w and int(w) > 200:
                    img_url = src
                    break

        if not img_url:
            return ""

        # Resolve relative URLs
        if img_url.startswith("/"):
            from urllib.parse import urljoin

            img_url = urljoin(article_url, img_url)

        img_resp = requests.get(img_url, timeout=15)
        img_resp.raise_for_status()

        # Validate it's actually an image
        content_type = img_resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return ""

        out_path.write_bytes(img_resp.content)
        print(f"    Saved image: {filename} ({len(img_resp.content):,} bytes)")
        return filename
    except Exception as exc:
        print(f"    Warning: failed to download image for article {idx}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# 2. LLM Analysis (OpenRouter)
# ---------------------------------------------------------------------------


def analyse_text(title: str, body: str) -> dict:
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
        "European Portuguese. You also act as a Hong Kong Cantonese teacher "
        "named Oreo Sir who teaches European Portuguese to Hong Kong A2-B1 "
        "students.\n\n"
        "Given a news article (title + body), perform the following tasks and "
        "return ONLY a strictly valid JSON object with NO markdown formatting.\n\n"
        "The JSON object must contain exactly these 5 top-level keys:\n"
        '1. "translation_en": a high-quality, natural English translation of the entire article (string).\n'
        '2. "translation_cn": a high-quality Traditional Chinese (繁體中文) translation of the entire article (string).\n'
        '3. "alignment": an array of 10-15 objects. Each object represents one short sentence from the article and must contain:\n'
        '    "pt" – the Portuguese sentence exactly as it appears in the text\n'
        '    "en" – the English translation of that sentence\n'
        '    "cn" – the Traditional Chinese translation of that sentence\n'
        '4. "analysis": an array of objects identifying CEFR B2/C1/C2 difficult words and conjugated verbs from the original Portuguese text. Each object contains:\n'
        '    "word"       – the word exactly as it appears in the Portuguese text\n'
        '    "infinitive" – the infinitive form if category is "verb", otherwise null\n'
        '    "en"         – concise English translation or explanation\n'
        '    "category"   – either "vocab" or "verb"\n'
        '5. "cantonese_teacher": (string) Act as Oreo Sir, a Hong Kong Cantonese teacher teaching European Portuguese to A2-B1 Hong Kong students. Write approximately 150 characters in vivid, authentic Hong Kong Cantonese (香港廣東話). The content should include: (a) what this news is about, and (b) 1-2 key A2/B1 vocabulary items or syntax points worth noting. Use romanisation for Portuguese words when needed.\n\n'
        "Rules:\n"
        "- Do NOT wrap the response in markdown code blocks.\n"
        "- Use double quotes for all strings.\n"
        "- Ensure the output is valid JSON.\n"
        "- Keep the alignment sentences short and natural (1-2 lines each)."
    )

    user_prompt = f"Title: {title}\n\nContent:\n{body}\n"

    print("[2/5] Analysing text with LLM (OpenRouter)…")
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

    # Normalise analysis items
    analysis = data.get("analysis", [])
    normalised: list[dict] = []
    if isinstance(analysis, list):
        for it in analysis:
            if not isinstance(it, dict):
                continue
            normalised.append(
                {
                    "word": str(it.get("word", "")),
                    "infinitive": (
                        it.get("infinitive")
                        if it.get("infinitive")
                        else None
                    ),
                    "en": str(it.get("en", "")),
                    "category": (
                        "verb"
                        if it.get("category") == "verb"
                        else "vocab"
                    ),
                }
            )

    alignment = data.get("alignment", [])
    if not isinstance(alignment, list):
        alignment = []

    print(f"    EN translation: {len(str(data.get('translation_en', '')))} chars")
    print(f"    CN translation: {len(str(data.get('translation_cn', '')))} chars")
    print(f"    Alignment: {len(alignment)} sentences")
    print(
        f"    Analysis: {len(normalised)} items "
        f"({sum(1 for i in normalised if i['category'] == 'verb')} verbs, "
        f"{sum(1 for i in normalised if i['category'] == 'vocab')} vocab)."
    )
    print(f"    Cantonese teacher: {len(str(data.get('cantonese_teacher', '')))} chars")

    return {
        "translation_en": str(data.get("translation_en", "")).strip(),
        "translation_cn": str(data.get("translation_cn", "")).strip(),
        "alignment": alignment,
        "analysis": normalised,
        "cantonese_teacher": str(data.get("cantonese_teacher", "")).strip(),
    }


# ---------------------------------------------------------------------------
# 3. Text-to-Speech (xAI Grok TTS)
# ---------------------------------------------------------------------------


def synthesise_speech(title: str, body: str, out_path: Path) -> None:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable XAI_API_KEY is not set.")

    # Send ONLY the original Portuguese text — never translations
    text = f"{title}. {body}"
    if len(text) > 14_000:
        text = text[:14_000]
        print("    Note: text truncated to 14,000 characters for TTS.")

    print("[3/5] Synthesising speech via xAI Grok TTS (voice=ara, lang=pt-PT)…")
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


def _apply_tooltips(text: str, items: list[dict]) -> str:
    """Escape text and highlight known words with tooltip spans."""
    lookup: dict[str, dict] = {}
    for it in items:
        w = it.get("word", "")
        if w:
            lookup[w.lower()] = it

    if not lookup:
        return html.escape(text)

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
    return pattern.sub(_replace, escaped_text)


def _build_alignment_html(
    alignment: list[dict], items: list[dict]
) -> tuple[str, str, str]:
    """
    Build PT, EN, and CN HTML using alignment sentences.
    PT sentences receive tooltip highlighting and sentence spans.
    """
    pt_parts: list[str] = []
    en_parts: list[str] = []
    cn_parts: list[str] = []

    for idx, sent in enumerate(alignment, start=1):
        pt_raw = str(sent.get("pt", ""))
        en_raw = str(sent.get("en", ""))
        cn_raw = str(sent.get("cn", ""))

        if not pt_raw.strip():
            continue

        pt_highlighted = _apply_tooltips(pt_raw, items)
        pt_parts.append(
            f'<span class="pt-sentence" data-id="{idx}">{pt_highlighted}</span>'
        )
        en_parts.append(
            f'<span class="en-sentence" data-id="{idx}">{html.escape(en_raw)}</span>'
        )
        cn_parts.append(
            f'<span class="cn-sentence" data-id="{idx}">{html.escape(cn_raw)}</span>'
        )

    pt_html = " ".join(pt_parts)
    en_html = " ".join(en_parts)
    cn_html = " ".join(cn_parts)
    return pt_html, en_html, cn_html


def generate_html_page(articles_data: list[dict], out_path: Path) -> None:
    print("[4/5] Generating HTML page…")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    article_blocks: list[str] = []
    for idx, article in enumerate(articles_data, start=1):
        alignment = article.get("alignment", [])
        items = article.get("analysis", [])

        if alignment:
            pt_html, en_html, cn_html = _build_alignment_html(alignment, items)
        else:
            # Fallback: no alignment – show plain highlighted body + full translations
            pt_html = _apply_tooltips(article["body"], items)
            en_html = html.escape(article.get("translation_en", ""))
            cn_html = html.escape(article.get("translation_cn", ""))

        oreo_text = html.escape(
            article.get("cantonese_teacher", "")
        ).replace("\n", "<br>")

        display = "block" if idx == 1 else "none"
        img_tag = ""
        if article.get("image"):
            img_tag = (
                f'<img src="{article["image"]}" '
                f'alt="Article image" class="article-img">\n    '
            )

        block = f"""<article class="card news-article" id="article-{idx}" style="display: {display};">
    {img_tag}<h1>{html.escape(article["title"])}</h1>
    <div class="meta">Notícias ao Minuto · PT-PT Daily News</div>
    <div class="oreo-block">
      <div class="oreo-header">👨‍🏫 我 Oreo Sir 話齋 (Cantonese Teacher)</div>
      <div class="oreo-content">{oreo_text}</div>
    </div>
    <audio controls>
      <source src="{article['audio']}" type="audio/mpeg">
      O seu navegador não suporta o elemento de áudio.
    </audio>
    <button class="btn-translation" data-target="transBox{idx}">Hide Translation</button>
    <div id="transBox{idx}" class="translation-box" style="display: block;">
      <div class="translation-col">
        <h3>English</h3>
        <div class="sentence-content">{en_html}</div>
      </div>
      <div class="translation-col">
        <h3>中文</h3>
        <div class="sentence-content">{cn_html}</div>
      </div>
    </div>
    <div class="content pt-content">
      <p>{pt_html}</p>
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
<title>PT-PT Daily News – Learn European Portuguese</title>
<style>
  :root {{
    --bg: #f4f6f8;
    --surface: #ffffff;
    --text: #1a1a1a;
    --muted: #6b7280;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --oreo-bg: #fff7ed;
    --oreo-border: #fb923c;
    --vocab-bg: #fef3c7;
    --vocab-fg: #92400e;
    --vocab-border: #f59e0b;
    --verb-bg: #fee2e2;
    --verb-fg: #991b1b;
    --verb-border: #ef4444;
    --sync-bg: #dbeafe;
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
    max-width: 860px;
    margin: 40px auto;
    padding: 0 20px;
  }}
  .card {{
    background: var(--surface);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 36px;
    margin-bottom: 24px;
  }}
  .article-img {{
    width: 100%;
    max-height: 360px;
    object-fit: cover;
    border-radius: 12px;
    margin-bottom: 20px;
  }}
  h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 10px;
    line-height: 1.3;
    letter-spacing: -0.02em;
  }}
  .meta {{
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 24px;
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
  .oreo-block {{
    background: var(--oreo-bg);
    border-left: 4px solid var(--oreo-border);
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    margin-bottom: 24px;
    color: #7c2d12;
  }}
  .oreo-header {{
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .oreo-content {{
    font-size: 0.95rem;
    line-height: 1.7;
  }}
  audio {{
    width: 100%;
    margin-bottom: 20px;
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
  .btn-translation:hover {{ background: #1d4ed8; }}
  .btn-translation:active {{ transform: scale(0.98); }}
  .translation-box {{
    background: var(--accent-light);
    border-left: 4px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 20px 22px;
    margin-bottom: 28px;
    color: #1e3a8a;
    line-height: 1.8;
    font-size: 0.97rem;
  }}
  .translation-box h3 {{
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 8px;
    color: #2563eb;
  }}
  .translation-col + .translation-col {{
    margin-top: 18px;
    padding-top: 18px;
    border-top: 1px solid #bfdbfe;
  }}
  .sentence-content {{
    text-align: justify;
  }}
  .pt-content p {{
    margin: 0;
    text-align: justify;
    font-size: 1.05rem;
    line-height: 2;
  }}
  .pt-sentence, .en-sentence, .cn-sentence {{
    cursor: default;
    transition: background 0.2s ease;
    padding: 2px 0;
    border-radius: 4px;
  }}
  .pt-sentence {{ display: inline; }}
  .pt-sentence:hover, .en-sentence:hover, .cn-sentence:hover {{
    background: var(--sync-bg);
  }}
  .sentence-highlight {{
    background: var(--sync-bg) !important;
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
  .tooltip.vocab:hover {{ background-color: #fde68a; }}
  .tooltip.verb {{
    background-color: var(--verb-bg);
    border-bottom: 2.5px solid var(--verb-border);
    color: var(--verb-fg);
  }}
  .tooltip.verb:hover {{ background-color: #fecaca; }}
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
    z-index: 30;
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
  #nextBtn:hover {{ background: #1d4ed8; }}
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
    .container {{ margin: 20px auto; }}
    .card {{ padding: 22px; }}
    h1 {{ font-size: 1.45rem; }}
    #nextBtn {{ font-size: 0.95rem; padding: 12px; }}
    .article-img {{ max-height: 240px; }}
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
    &nbsp;&middot;&nbsp;
    <span class="legend">
      <span class="dot" style="background:#dbeafe;"></span> Sentence sync
    </span>
    <br><br>
    Generated automatically by PT-PT Daily News
  </footer>
</div>
<script>
(function() {{
  // Toggle translation visibility per article
  document.querySelectorAll('.btn-translation').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var targetId = btn.getAttribute('data-target');
      var box = document.getElementById(targetId);
      if (box) {{
        if (box.style.display === 'none') {{
          box.style.display = 'block';
          btn.textContent = 'Hide Translation';
        }} else {{
          box.style.display = 'none';
          btn.textContent = 'Show Translation';
        }}
      }}
    }});
  }});

  // Sentence hover sync across PT / EN / CN
  function highlightGroup(id) {{
    document.querySelectorAll('[data-id="' + id + '"]').forEach(function(el) {{
      el.classList.add('sentence-highlight');
    }});
  }}
  function unhighlightGroup(id) {{
    document.querySelectorAll('[data-id="' + id + '"]').forEach(function(el) {{
      el.classList.remove('sentence-highlight');
    }});
  }}
  document.querySelectorAll('.pt-sentence, .en-sentence, .cn-sentence').forEach(function(span) {{
    span.addEventListener('mouseenter', function() {{
      var id = this.getAttribute('data-id');
      if (id) highlightGroup(id);
    }});
    span.addEventListener('mouseleave', function() {{
      var id = this.getAttribute('data-id');
      if (id) unhighlightGroup(id);
    }});
  }});

  // Next News navigation
  var current = 1;
  var total = {total_articles};
  var nextBtn = document.getElementById('nextBtn');
  if (nextBtn) {{
    nextBtn.addEventListener('click', function() {{
      if (current >= total) return;
      var cur = document.getElementById('article-' + current);
      if (cur) cur.style.display = 'none';
      current++;
      var nxt = document.getElementById('article-' + current);
      if (nxt) {{
        nxt.style.display = 'block';
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
# 5. Email Digest
# ---------------------------------------------------------------------------


def send_email_digest(articles: list[dict]) -> None:
    smtp_server = os.environ.get("SMTP_SERVER")
    email_user = os.environ.get("EMAIL_USER")
    email_pass = os.environ.get("EMAIL_PASS")

    if not all([smtp_server, email_user, email_pass]):
        print("[5/5] SMTP credentials not set, skipping email.")
        return

    print("[5/5] Sending email digest…")

    recipients = ["trevorlau@gmail.com", "claire.cheukying@gmail.com"]
    subject = "🇵🇹 PT-PT Daily News – Morning Brief"

    # Build article list
    article_list_items = ""
    for i, art in enumerate(articles, start=1):
        article_list_items += f"<li><strong>{html.escape(art['title'])}</strong></li>\n"

    oreo_summary = html.escape(articles[0].get("cantonese_teacher", "N/A"))

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.7; color: #1a1a1a; max-width: 600px; margin: 0 auto; padding: 20px;">
  <h2 style="color: #2563eb;">🇵🇹 PT-PT Daily News – Morning Brief</h2>
  <p>Good morning! Here are today's 3 European Portuguese news picks:</p>
  <ol>
    {article_list_items}
  </ol>
  <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;">
  <h3 style="color: #fb923c;">👨‍🏫 Oreo Sir's Cantonese Summary (Article 1)</h3>
  <div style="background: #fff7ed; border-left: 4px solid #fb923c; padding: 16px 20px; border-radius: 0 8px 8px 0; color: #7c2d12;">
    {oreo_summary}
  </div>
  <p style="margin-top: 28px;">
    <a href="{html.escape(GITHUB_PAGES_URL)}" style="display: inline-block; background: #2563eb; color: #fff; text-decoration: none; padding: 12px 20px; border-radius: 8px; font-weight: 600;">
      📖 Read full articles on GitHub Pages
    </a>
  </p>
  <footer style="margin-top: 32px; font-size: 0.85rem; color: #6b7280; text-align: center;">
    Generated automatically by PT-PT Daily News
  </footer>
</body>
</html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = ", ".join(recipients)

    # Plain-text fallback
    plain_body = (
        f"PT-PT Daily News – Morning Brief\n\n"
        f"Today's articles:\n"
        + "\n".join(f"  • {art['title']}" for art in articles)
        + f"\n\nOreo Sir's Summary:\n{articles[0].get('cantonese_teacher', 'N/A')}\n\n"
        f"Read more: {GITHUB_PAGES_URL}\n"
    )

    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(smtp_server, 587, timeout=30) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.sendmail(email_user, recipients, msg.as_string())
        print(f"    Email sent to {', '.join(recipients)}")
    except Exception as exc:
        print(f"    Warning: failed to send email: {exc}")


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

            # Download image
            img_filename = fetch_article_image(article, i, public_html)
            article["image"] = img_filename

            # LLM analysis
            result = analyse_text(article["title"], article["body"])
            article["translation_en"] = result["translation_en"]
            article["translation_cn"] = result["translation_cn"]
            article["alignment"] = result["alignment"]
            article["analysis"] = result["analysis"]
            article["cantonese_teacher"] = result["cantonese_teacher"]

            # TTS
            audio_filename = f"news_{i}.mp3"
            mp3_path = public_html / audio_filename
            synthesise_speech(article["title"], article["body"], mp3_path)
            article["audio"] = audio_filename

        # Generate HTML
        html_path = public_html / "index.html"
        generate_html_page(articles, html_path)

        # Send email
        send_email_digest(articles)

        print("\n✅  Done! Open public_html/index.html in your browser.")
        return 0
    except Exception as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
