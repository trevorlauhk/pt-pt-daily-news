#!/usr/bin/env python3
"""
PT-PT Daily News – Dual-Engine AI Learning Platform

Generates two daily Portuguese learning articles via AI:
  1. A2 Level (CIPLE) – simple tenses, ~200 words
  2. B1-B2 Level (DEPLE/DIPLE) – subjunctive, complex clauses, ~250-300 words

Analyses vocabulary with OpenRouter (Mistral), synthesises speech with
xAI Grok TTS, renders an interactive dual-tab HTML page with
sentence-sync hover & tooltips, and sends a morning email digest.

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
import random
import re
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from openai import OpenAI

GITHUB_PAGES_URL = os.environ.get(
    "GITHUB_PAGES_URL", "https://yourname.github.io/pt-pt-daily-news"
)

# ---------------------------------------------------------------------------
# Topic Banks
# ---------------------------------------------------------------------------

CIPLE_A2_TOPICS = [
    "Pedir comida num restaurante (在餐廳點餐)",
    "Alugar um apartamento (租公寓)",
    "Ir ao médico (看醫生)",
    "Ir aos CTT / correios (去郵局)",
    "Comprar bilhetes de comboio (買火車票)",
    "Abrir uma conta bancária (在銀行開戶)",
    "Ir ao cabeleireiro (去理髮店)",
    "Fazer compras no supermercado (超市購物)",
    "Pedir informações na rua (問路)",
    "Comprar medicamentos na farmácia (在藥房買藥)",
    "Marcar consulta no dentista (預約牙醫)",
    "Ir ao ginásio (去健身房)",
    "Enviar uma encomenda (寄包裹)",
    "Num café com um amigo (在咖啡店見朋友)",
    "Contratar um tarifário de telemóvel (買手機套餐)",
    "Chamar a polícia / pedir ajuda (報警求助)",
    "Requisitar livros na biblioteca (在圖書館借書)",
    "Ir à lavandaria (去洗衣店)",
    "Levar o carro à oficina (修車)",
    "Fazer check-in no aeroporto (機場登機手續)",
    "Visitar um apartamento para alugar (租房看房)",
    "Na pastelaria (在麵包店)",
    "Contratar um seguro (買保險)",
    "Ir a uma festa de aniversário (參加生日派對)",
    "Ir à Câmara Municipal (去市政廳辦事)",
    "Visitar o jardim zoológico (去動物園)",
    "Comprar roupa numa loja (買衣服)",
    "Ir à praia (去海灘)",
    "Abastecer o carro (在加油站加油)",
    "Chamar um canalizador (預約修水管)",
]

B1_B2_TOPICS = [
    "O impacto do turismo em Portugal (旅遊業的影響)",
    "Vantagens e desvantagens do trabalho remoto (遠程工作的利弊)",
    "Sustentabilidade e alterações climáticas (可持續發展與氣候變化)",
    "Reclamar sobre um contrato de telecomunicações (處理電訊合同投訴)",
    "A importância da arte e da cultura na sociedade (藝術文化的重要性)",
    "Diferenças entre viver na cidade e no campo (城鄉生活差異)",
    "A influência das redes sociais nos jovens (社交媒體對年輕人的影響)",
    "Falar sobre um evento histórico de Portugal (談論葡萄牙歷史事件)",
    "Debate sobre a semana de trabalho de 4 dias (4天工作制辯論)",
    "Expressar opinião sobre o uso de Inteligência Artificial (對AI的看法)",
    "A crise da habitação em Lisboa e no Porto (里斯本與波圖的住屋危機)",
    "O futuro da energia renovável em Portugal (葡萄牙可再生能源的未來)",
    "Imigração e integração multicultural (移民與多元文化融合)",
    "O papel da educação no século XXI (21世紀教育的角色)",
    "A evolução da gastronomia portuguesa (葡萄牙美食的演變)",
    "Desafios do sistema de saúde nacional (國家醫療系統的挑戰)",
    "O impacto da criptomoeda na economia (加密貨幣對經濟的影響)",
    "Liberdade de expressão vs. desinformação online (言論自由與網絡假資訊)",
    "O valor do património histórico e arquitetónico (歷史與建築遺產的價值)",
    "Mobilidade urbana e transportes públicos (城市流動性與公共交通)",
]


# ---------------------------------------------------------------------------
# 1. Topic Selection
# ---------------------------------------------------------------------------


def pick_daily_topics() -> tuple[str, str]:
    a2_topic = random.choice(CIPLE_A2_TOPICS)
    b1b2_topic = random.choice(B1_B2_TOPICS)
    print(f"[1/5] Today's A2 topic:  {a2_topic}")
    print(f"       Today's B1-B2 topic: {b1b2_topic}")
    return a2_topic, b1b2_topic


# ---------------------------------------------------------------------------
# 2. LLM Story Generation (OpenRouter)
# ---------------------------------------------------------------------------


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
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

    response = client.chat.completions.create(
        model="mistralai/mistral-large-2407",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
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

    return data


def generate_a2_story(topic: str) -> dict:
    system_prompt = (
        "You are a European Portuguese language teacher preparing material for "
        "the CIPLE (A2) level exam.\n\n"
        "Write a short, natural-sounding story or dialogue about the topic provided.\n"
        "Requirements:\n"
        "- Strictly A2 (CIPLE) level.\n"
        "- Use ONLY simple tenses: Presente do Indicativo, Pretérito Perfeito, "
        "Pretérito Imperfeito, and Futuro Próximo (ir + infinitivo).\n"
        "- Avoid subjunctive, conditional, or complex subordinate clauses.\n"
        "- Use basic connectors (e, mas, porque, então, depois, antes, também).\n"
        "- Length: approximately 180-220 words.\n"
        "- The text should feel like a real short story or diary entry, NOT a list of sentences.\n\n"
        "Return ONLY a strictly valid JSON object with NO markdown formatting, containing exactly:\n"
        '  "title": a catchy Portuguese title for the story (string)\n'
        '  "story_pt": the full Portuguese story (string)\n'
        '  "translation_en": a natural English translation of the entire story (string)\n'
        '  "translation_cn": a natural Traditional Chinese (繁體中文) translation (string)\n'
        '  "alignment": an array of 10-15 objects, each with keys "pt", "en", "cn" representing aligned short sentences (1-2 lines each).\n'
        '  "analysis": an array of 8-12 vocabulary/grammar objects from the Portuguese text, each with:\n'
        '    "word" – the word exactly as it appears\n'
        '    "infinitive" – infinitive if it is a conjugated verb, otherwise null\n'
        '    "en" – concise English explanation\n'
        '    "category" – "vocab" or "verb"\n\n'
        "Rules:\n"
        "- Do NOT wrap the response in markdown code blocks.\n"
        "- Use double quotes for all strings.\n"
        "- Ensure valid JSON."
    )

    user_prompt = f"Topic: {topic}\n\nPlease write the A2 story and return the JSON object."

    print("[2/5] Generating A2 story with LLM…")
    data = _call_llm(system_prompt, user_prompt)
    return _normalise_article(data, "A2")


def generate_b1b2_story(topic: str) -> dict:
    system_prompt = (
        "You are a European Portuguese language teacher preparing material for "
        "the DEPLE (B1) / DIPLE (B2) level exams.\n\n"
        "Write a thoughtful article, opinion piece, or narrative about the topic provided.\n"
        "Requirements:\n"
        "- B1-B2 level. Vocabulary can be more abstract and nuanced.\n"
        "- MUST use the Subjunctive (Conjuntivo) mood at least 3 times.\n"
        "- MUST use complex subordinate clauses (e.g. relative clauses, concessive clauses).\n"
        "- MUST use advanced connectors: portanto, contudo, no entanto, por conseguinte, "
        "apesar de, uma vez que, desde que, a fim de que, etc.\n"
        "- You may use all indicative tenses plus conditional and subjunctive.\n"
        "- Length: approximately 250-300 words.\n"
        "- The text should read like a real newspaper opinion piece or feature article.\n\n"
        "Return ONLY a strictly valid JSON object with NO markdown formatting, containing exactly:\n"
        '  "title": a catchy Portuguese title for the article (string)\n'
        '  "story_pt": the full Portuguese article (string)\n'
        '  "translation_en": a natural English translation (string)\n'
        '  "translation_cn": a natural Traditional Chinese (繁體中文) translation (string)\n'
        '  "alignment": an array of 12-18 objects, each with keys "pt", "en", "cn" representing aligned short sentences (1-2 lines each).\n'
        '  "analysis": an array of 10-15 vocabulary/grammar objects from the Portuguese text, each with:\n'
        '    "word" – the word exactly as it appears\n'
        '    "infinitive" – infinitive if it is a conjugated verb, otherwise null\n'
        '    "en" – concise English explanation\n'
        '    "category" – "vocab" or "verb"\n\n'
        "Rules:\n"
        "- Do NOT wrap the response in markdown code blocks.\n"
        "- Use double quotes for all strings.\n"
        "- Ensure valid JSON."
    )

    user_prompt = f"Topic: {topic}\n\nPlease write the B1-B2 article and return the JSON object."

    print("[3/5] Generating B1-B2 story with LLM…")
    data = _call_llm(system_prompt, user_prompt)
    return _normalise_article(data, "B1-B2")


def _normalise_article(data: dict, level: str) -> dict:
    """Validate and normalise the LLM response."""
    alignment = data.get("alignment", [])
    if not isinstance(alignment, list):
        alignment = []

    analysis = data.get("analysis", [])
    normalised_analysis: list[dict] = []
    if isinstance(analysis, list):
        for it in analysis:
            if not isinstance(it, dict):
                continue
            normalised_analysis.append(
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

    article = {
        "level": level,
        "title": str(data.get("title", "")).strip(),
        "story_pt": str(data.get("story_pt", "")).strip(),
        "translation_en": str(data.get("translation_en", "")).strip(),
        "translation_cn": str(data.get("translation_cn", "")).strip(),
        "alignment": alignment,
        "analysis": normalised_analysis,
    }

    print(f"    Title: {article['title'][:70]}{'…' if len(article['title']) > 70 else ''}")
    print(f"    PT length: {len(article['story_pt'])} chars")
    print(f"    Alignment: {len(alignment)} sentences")
    print(f"    Analysis: {len(normalised_analysis)} items")
    return article


# ---------------------------------------------------------------------------
# 3. Text-to-Speech (xAI Grok TTS)
# ---------------------------------------------------------------------------


def synthesise_speech(text: str, out_path: Path) -> None:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable XAI_API_KEY is not set.")

    if len(text) > 14_000:
        text = text[:14_000]
        print("    Note: text truncated to 14,000 characters for TTS.")

    print(f"    Synthesising speech via xAI Grok TTS (voice=ara, lang=pt-PT)…")
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


def _build_alignment_html(alignment: list[dict], items: list[dict]) -> tuple[str, str, str]:
    """Build PT, EN, and CN HTML using alignment sentences."""
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

    return " ".join(pt_parts), " ".join(en_parts), " ".join(cn_parts)


def _render_article_card(article: dict, article_id: str, display: str) -> str:
    """Render a single article card HTML."""
    alignment = article.get("alignment", [])
    items = article.get("analysis", [])

    if alignment:
        pt_html, en_html, cn_html = _build_alignment_html(alignment, items)
    else:
        pt_html = _apply_tooltips(article["story_pt"], items)
        en_html = html.escape(article.get("translation_en", ""))
        cn_html = html.escape(article.get("translation_cn", ""))

    level_badge = (
        '<span class="level-badge a2">A2 (CIPLE)</span>'
        if article["level"] == "A2"
        else '<span class="level-badge b1b2">B1-B2 (DEPLE/DIPLE)</span>'
    )

    return f"""<article class="card news-article" id="{article_id}" style="display: {display};">
    <div class="article-header">
      <h1>{html.escape(article["title"])}</h1>
      {level_badge}
    </div>
    <audio controls>
      <source src="{article['audio']}" type="audio/mpeg">
      O seu navegador não suporta o elemento de áudio.
    </audio>
    <button class="btn-translation" data-target="transBox-{article_id}">Hide Translation</button>
    <div id="transBox-{article_id}" class="translation-box" style="display: block;">
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


def generate_html_page(articles: list[dict], out_path: Path) -> None:
    print("[5/5] Generating HTML page…")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    a2_article = next((a for a in articles if a["level"] == "A2"), articles[0])
    b1b2_article = next((a for a in articles if a["level"] == "B1-B2"), articles[-1])

    a2_card = _render_article_card(a2_article, "article-a2", "block")
    b1b2_card = _render_article_card(b1b2_article, "article-b1b2", "none")

    page = f"""<!DOCTYPE html>
<html lang="pt-PT">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PT-PT Daily News – Dual-Engine Learning</title>
<style>
  :root {{
    --bg: #f4f6f8;
    --surface: #ffffff;
    --text: #1a1a1a;
    --muted: #6b7280;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --a2-color: #059669;
    --a2-bg: #d1fae5;
    --b1b2-color: #7c3aed;
    --b1b2-bg: #ede9fe;
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
    margin: 0 auto;
    padding: 0 20px 40px;
  }}
  /* Tabs */
  .tab-bar {{
    display: flex;
    gap: 0;
    margin: 24px 0;
    background: var(--surface);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    padding: 6px;
  }}
  .tab-btn {{
    flex: 1;
    padding: 14px;
    font-size: 1rem;
    font-weight: 600;
    border: none;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    border-radius: 10px;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }}
  .tab-btn:hover {{ background: #f3f4f6; }}
  .tab-btn.active-a2 {{
    background: var(--a2-bg);
    color: var(--a2-color);
    box-shadow: 0 2px 8px rgba(5,150,105,0.15);
  }}
  .tab-btn.active-b1b2 {{
    background: var(--b1b2-bg);
    color: var(--b1b2-color);
    box-shadow: 0 2px 8px rgba(124,58,237,0.15);
  }}
  .tab-badge {{
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 700;
  }}
  .tab-btn.active-a2 .tab-badge {{
    background: var(--a2-color);
    color: #fff;
  }}
  .tab-btn.active-b1b2 .tab-badge {{
    background: var(--b1b2-color);
    color: #fff;
  }}
  /* Cards */
  .card {{
    background: var(--surface);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 36px;
  }}
  .article-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 4px;
  }}
  h1 {{
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.3;
    letter-spacing: -0.02em;
    flex: 1;
  }}
  .level-badge {{
    font-size: 0.78rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 20px;
    white-space: nowrap;
    margin-top: 6px;
  }}
  .level-badge.a2 {{
    background: var(--a2-bg);
    color: var(--a2-color);
  }}
  .level-badge.b1b2 {{
    background: var(--b1b2-bg);
    color: var(--b1b2-color);
  }}
  audio {{
    width: 100%;
    margin: 20px 0;
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
    .container {{ padding: 0 16px 32px; }}
    .card {{ padding: 22px; }}
    h1 {{ font-size: 1.35rem; }}
    .tab-btn {{ font-size: 0.9rem; padding: 12px; }}
    .article-header {{ flex-direction: column; }}
    .level-badge {{ margin-top: 0; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="tab-bar">
    <button id="tab-a2" class="tab-btn active-a2" onclick="switchTab('A2')">
      🇵🇹 A2 Level <span class="tab-badge">CIPLE</span>
    </button>
    <button id="tab-b1b2" class="tab-btn" onclick="switchTab('B1-B2')">
      🇵🇹 B1-B2 Level <span class="tab-badge">DEPLE/DIPLE</span>
    </button>
  </div>

  {a2_card}
  {b1b2_card}

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
    PT-PT Daily News – Dual-Engine Learning Platform
  </footer>
</div>
<script>
(function() {{
  // Tab switching
  window.switchTab = function(level) {{
    var a2Card = document.getElementById('article-a2');
    var b1b2Card = document.getElementById('article-b1b2');
    var tabA2 = document.getElementById('tab-a2');
    var tabB1B2 = document.getElementById('tab-b1b2');

    if (level === 'A2') {{
      a2Card.style.display = 'block';
      b1b2Card.style.display = 'none';
      tabA2.classList.add('active-a2');
      tabB1B2.classList.remove('active-b1b2');
    }} else {{
      a2Card.style.display = 'none';
      b1b2Card.style.display = 'block';
      tabA2.classList.remove('active-a2');
      tabB1B2.classList.add('active-b1b2');
    }}
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }};

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
    subject = "🇵🇹 PT-PT Daily News – Dual-Engine Learning"

    a2 = next((a for a in articles if a["level"] == "A2"), articles[0])
    b1b2 = next((a for a in articles if a["level"] == "B1-B2"), articles[-1])

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.7; color: #1a1a1a; max-width: 600px; margin: 0 auto; padding: 20px;">
  <h2 style="color: #2563eb;">🇵🇹 PT-PT Daily News – Dual-Engine Learning</h2>
  <p>Good morning! Here are today's AI-generated Portuguese learning articles:</p>

  <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;">

  <div style="background: #d1fae5; border-left: 4px solid #059669; padding: 16px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
    <h3 style="color: #059669; margin: 0 0 8px;">📗 A2 Level (CIPLE)</h3>
    <p style="margin: 0; color: #064e3b;"><strong>{html.escape(a2['title'])}</strong></p>
    <p style="margin: 8px 0 0; font-size: 0.9rem; color: #065f46;">{html.escape(a2.get('topic', ''))}</p>
  </div>

  <div style="background: #ede9fe; border-left: 4px solid #7c3aed; padding: 16px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
    <h3 style="color: #7c3aed; margin: 0 0 8px;">📘 B1-B2 Level (DEPLE/DIPLE)</h3>
    <p style="margin: 0; color: #4c1d95;"><strong>{html.escape(b1b2['title'])}</strong></p>
    <p style="margin: 8px 0 0; font-size: 0.9rem; color: #5b21b6;">{html.escape(b1b2.get('topic', ''))}</p>
  </div>

  <p style="margin-top: 28px;">
    <a href="{html.escape(GITHUB_PAGES_URL)}" style="display: inline-block; background: #2563eb; color: #fff; text-decoration: none; padding: 12px 20px; border-radius: 8px; font-weight: 600;">
      📖 Read full articles on GitHub Pages
    </a>
  </p>

  <footer style="margin-top: 32px; font-size: 0.85rem; color: #6b7280; text-align: center;">
    Generated automatically by PT-PT Daily News – Dual-Engine Learning
  </footer>
</body>
</html>
"""

    # Plain-text fallback
    plain_body = (
        f"PT-PT Daily News – Dual-Engine Learning\n\n"
        f"Today's A2 Article (CIPLE):\n"
        f"  {a2['title']}\n"
        f"  Topic: {a2.get('topic', 'N/A')}\n\n"
        f"Today's B1-B2 Article (DEPLE/DIPLE):\n"
        f"  {b1b2['title']}\n"
        f"  Topic: {b1b2.get('topic', 'N/A')}\n\n"
        f"Read more: {GITHUB_PAGES_URL}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = ", ".join(recipients)

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
        a2_topic, b1b2_topic = pick_daily_topics()

        public_html = Path("public_html")
        public_html.mkdir(parents=True, exist_ok=True)

        # Generate A2 story
        print("\n--- Generating A2 (CIPLE) article ---")
        a2_article = generate_a2_story(a2_topic)
        a2_article["topic"] = a2_topic
        a2_article["level"] = "A2"

        # Generate B1-B2 story
        print("\n--- Generating B1-B2 (DEPLE/DIPLE) article ---")
        b1b2_article = generate_b1b2_story(b1b2_topic)
        b1b2_article["topic"] = b1b2_topic
        b1b2_article["level"] = "B1-B2"

        articles = [a2_article, b1b2_article]

        # TTS for both
        print("\n--- Synthesising speech ---")
        synthesise_speech(a2_article["story_pt"], public_html / "news_a2.mp3")
        a2_article["audio"] = "news_a2.mp3"

        synthesise_speech(b1b2_article["story_pt"], public_html / "news_b1b2.mp3")
        b1b2_article["audio"] = "news_b1b2.mp3"

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
