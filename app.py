#!/usr/bin/env python3
"""
🍪 Oreo Sir 的葡文補習社 – Streamlit Interactive App

A local interactive Portuguese learning web app powered by:
  • OpenRouter (Mistral) for AI-generated stories & explanations
  • Azure TTS (pt-PT-FernandaNeural / pt-PT-DuarteNeural) for speech

Usage:
    streamlit run app.py

Required env vars (in .env file):
    OPENROUTER_API_KEY
    AZURE_TTS_KEY

Required packages:
    pip install streamlit python-dotenv openai requests
"""

from __future__ import annotations

import html
import json
import os
import random
import re
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Topic Banks (reused from main.py)
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
# Core AI & TTS functions (reused from main.py)
# ---------------------------------------------------------------------------


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Call OpenRouter (Mistral) and return raw text response."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("❌ 未設置 OPENROUTER_API_KEY！請在 .env 文件中添加。")
        return ""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Oreo Sir PT-PT",
        },
    )

    response = client.chat.completions.create(
        model="mistralai/mistral-large-2407",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


def _call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> dict:
    """Call OpenRouter (Mistral) and return parsed JSON dict."""
    raw = _call_llm(system_prompt, user_prompt, temperature)
    if not raw:
        return {}

    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError as exc:
        st.error(f"❌ JSON 解析失敗: {exc}")
        st.code(raw, language="json")
        return {}


def synthesise_speech(text: str, voice_name: str = "pt-PT-DuarteNeural") -> bytes:
    """Call Azure TTS and return MP3 audio bytes."""
    api_key = os.environ.get("AZURE_TTS_KEY")
    if not api_key:
        st.error("❌ 未設置 AZURE_TTS_KEY！請在 .env 文件中添加。")
        return b""

    region = "eastasia"
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        "User-Agent": "PTPTDailyNews",
    }

    escaped_text = html.escape(text)
    ssml = f"""<speak version='1.0' xml:lang='pt-PT'>
        <voice xml:lang='pt-PT' name='{voice_name}'>
            {escaped_text}
        </voice>
    </speak>"""

    resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=120)
    if resp.status_code != 200:
        st.error(f"❌ Azure TTS 錯誤 {resp.status_code}: {resp.text[:200]}")
        return b""

    return resp.content


def _normalise_article(data: dict, level: str) -> dict:
    """Validate and normalise the LLM response."""
    alignment = data.get("alignment", [])
    if not isinstance(alignment, list):
        alignment = []

    analysis = data.get("analysis", [])
    normalised_analysis = []
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

    return {
        "level": level,
        "title": str(data.get("title", "")).strip(),
        "story_pt": str(data.get("story_pt", "")).strip(),
        "translation_en": str(data.get("translation_en", "")).strip(),
        "translation_cn": str(data.get("translation_cn", "")).strip(),
        "alignment": alignment,
        "analysis": normalised_analysis,
        "oreo_tips": str(data.get("oreo_tips", "")).strip(),
    }


def generate_a2_story() -> dict:
    topic = random.choice(CIPLE_A2_TOPICS)
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
        '    "category" – "vocab" or "verb"\n'
        '  "oreo_tips": (string) Write as "Oreo Sir", a friendly, humorous dog mascot teaching Portuguese to Hong Kong students. Use Traditional Chinese (繁體中文). Include at least 2 dog phrases/puns (汪星人用語) and 1-2 cookie 🍪 emojis. Write a short, warm grammar or vocabulary tip specifically for this article (around 80-120 characters). Keep it fun and encouraging, like a dog cheering you on!\n\n'
        "Rules:\n"
        "- Do NOT wrap the response in markdown code blocks.\n"
        "- Use double quotes for all strings.\n"
        "- Ensure valid JSON."
    )

    user_prompt = f"Topic: {topic}\n\nPlease write the A2 story and return the JSON object."
    data = _call_llm_json(system_prompt, user_prompt)
    article = _normalise_article(data, "A2")
    article["topic"] = topic
    return article


def generate_b1b2_story() -> dict:
    topic = random.choice(B1_B2_TOPICS)
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
        '    "category" – "vocab" or "verb"\n'
        '  "oreo_tips": (string) Write as "Oreo Sir", a friendly, humorous dog mascot teaching Portuguese to Hong Kong students. Use Traditional Chinese (繁體中文). Include at least 2 dog phrases/puns (汪星人用語) and 1-2 cookie 🍪 emojis. Write a short, warm grammar or vocabulary tip specifically for this article (around 100-140 characters). The tip should be slightly more advanced (B1-B2 level), maybe mentioning conjuntivo or complex connectors. Keep it fun and encouraging, like a wise dog cheering you on!\n\n'
        "Rules:\n"
        "- Do NOT wrap the response in markdown code blocks.\n"
        "- Use double quotes for all strings.\n"
        "- Ensure valid JSON."
    )

    user_prompt = f"Topic: {topic}\n\nPlease write the B1-B2 article and return the JSON object."
    data = _call_llm_json(system_prompt, user_prompt)
    article = _normalise_article(data, "B1-B2")
    article["topic"] = topic
    return article


def ask_oreo_sir(portuguese_text: str) -> str:
    """Send user Portuguese text to Mistral and get Oreo Sir's analysis."""
    system_prompt = (
        "你是一隻叫 Oreo Sir 的聰明狗狗，是專教香港人葡萄牙文的老師。"
        "請用繁體中文和狗星人語氣（加入 🍪 和 🐶 emoji）。"
        "請幫我："
        "1. 翻譯這段文字。"
        "2. 揪出最難的 3-5 個單字或動詞（標註原型和時態，特別留意 Conjuntivo 虛擬式）。"
        "3. 用簡單幽默的方式解釋文法重點。"
    )
    return _call_llm(system_prompt, portuguese_text, temperature=0.7)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🍪 Oreo Sir 的葡文補習社",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🍪 Oreo Sir 的葡文補習社")
st.caption("🇵🇹 專業歐洲葡萄牙語學習平台 · Powered by Mistral & Azure TTS")

# Session state init
for key in ["a2_article", "b1b2_article", "a2_audio", "b1b2_audio"]:
    if key not in st.session_state:
        st.session_state[key] = None

tab1, tab2 = st.tabs(["📰 今日雙引擎報紙", "🐶 貼文問 Oreo Sir"])


# ==================== TAB 1 ====================
with tab1:
    st.header("📰 今日雙引擎報紙")
    st.markdown(
        "點擊下方按鈕，Oreo Sir 會幫你生成 **A2 (CIPLE)** 和 **B1-B2 (DEPLE/DIPLE)** 兩篇文章，並配上真人發音！"
    )

    if st.button("🎲 生成今日新聞", type="primary", use_container_width=True):
        progress_text = st.empty()
        bar = st.progress(0)

        # A2
        progress_text.write("📝 正在生成 A2 (CIPLE) 文章...")
        a2_article = generate_a2_story()
        if a2_article and a2_article.get("story_pt"):
            st.session_state.a2_article = a2_article
            st.session_state.a2_audio = synthesise_speech(
                a2_article["story_pt"], voice_name="pt-PT-FernandaNeural"
            )
            bar.progress(50)
            progress_text.write("✅ A2 文章完成！正在生成 B1-B2...")
        else:
            st.error("❌ A2 文章生成失敗")
            st.stop()

        # B1-B2
        b1b2_article = generate_b1b2_story()
        if b1b2_article and b1b2_article.get("story_pt"):
            st.session_state.b1b2_article = b1b2_article
            st.session_state.b1b2_audio = synthesise_speech(
                b1b2_article["story_pt"], voice_name="pt-PT-DuarteNeural"
            )
            bar.progress(100)
            progress_text.write("🎉 兩篇文章全部完成！")
        else:
            st.error("❌ B1-B2 文章生成失敗")
            st.stop()

        st.success("🎉 今日新聞已生成！向下滾動查看內容 👇")
        st.balloons()

    # Display A2
    if st.session_state.a2_article:
        article = st.session_state.a2_article
        with st.container():
            st.markdown("---")
            st.subheader(f"📗 A2 Level (CIPLE) – {article['title']}")
            st.caption(f"📝 主題: {article.get('topic', '')}")

            if article.get("oreo_tips"):
                st.info(f"🐶 **Oreo Sir 小 Tips**\n\n{article['oreo_tips']}")

            if st.session_state.a2_audio:
                st.audio(st.session_state.a2_audio, format="audio/mp3")

            st.markdown("#### 📖 三語對照學習")
            if article.get("alignment"):
                for i, sent in enumerate(article["alignment"]):
                    cols = st.columns(3)
                    cols[0].markdown(f"**🇵🇹 葡文**\n\n{sent.get('pt', '')}")
                    cols[1].markdown(f"**🇬🇧 英文**\n\n{sent.get('en', '')}")
                    cols[2].markdown(f"**🇹🇼 中文**\n\n{sent.get('cn', '')}")
                    st.markdown("---")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🇵🇹 原文**\n\n{article['story_pt']}")
                with col2:
                    st.markdown(f"**🇬🇧 英文翻譯**\n\n{article['translation_en']}")
                    st.markdown(f"**🇹🇼 中文翻譯**\n\n{article['translation_cn']}")

    # Display B1-B2
    if st.session_state.b1b2_article:
        article = st.session_state.b1b2_article
        with st.container():
            st.markdown("---")
            st.subheader(f"📘 B1-B2 Level (DEPLE/DIPLE) – {article['title']}")
            st.caption(f"📝 主題: {article.get('topic', '')}")

            if article.get("oreo_tips"):
                st.info(f"🐶 **Oreo Sir 小 Tips**\n\n{article['oreo_tips']}")

            if st.session_state.b1b2_audio:
                st.audio(st.session_state.b1b2_audio, format="audio/mp3")

            st.markdown("#### 📖 三語對照學習")
            if article.get("alignment"):
                for i, sent in enumerate(article["alignment"]):
                    cols = st.columns(3)
                    cols[0].markdown(f"**🇵🇹 葡文**\n\n{sent.get('pt', '')}")
                    cols[1].markdown(f"**🇬🇧 英文**\n\n{sent.get('en', '')}")
                    cols[2].markdown(f"**🇹🇼 中文**\n\n{sent.get('cn', '')}")
                    st.markdown("---")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🇵🇹 原文**\n\n{article['story_pt']}")
                with col2:
                    st.markdown(f"**🇬🇧 英文翻譯**\n\n{article['translation_en']}")
                    st.markdown(f"**🇹🇼 中文翻譯**\n\n{article['translation_cn']}")


# ==================== TAB 2 ====================
with tab2:
    st.header("🐶 貼文問 Oreo Sir")
    st.markdown(
        "貼上任何你看不懂的葡萄牙文句子或文章，Oreo Sir 會幫你 **咬碎** 它！"
    )

    user_input = st.text_area(
        "📋 貼上你的葡萄牙文",
        placeholder="例如：Apesar de ter chovido ontem, decidimos ir à praia porque o tempo melhorou de repente.",
        height=150,
    )

    if st.button("🐾 咬碎這段文字！", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ 請先貼上一些葡萄牙文！")
        else:
            col_audio, col_analysis = st.columns([1, 2])

            with col_audio:
                st.markdown("### 🔊 原文發音")
                with st.spinner("🔊 生成發音中..."):
                    audio_bytes = synthesise_speech(
                        user_input, voice_name="pt-PT-DuarteNeural"
                    )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

            with col_analysis:
                st.markdown("### 🐶 Oreo Sir 的解析")
                with st.spinner("🐶 Oreo Sir 正在咬文嚼字..."):
                    analysis = ask_oreo_sir(user_input)
                if analysis:
                    st.markdown(analysis)

st.markdown("---")
st.caption("Made with 🍪 by Oreo Sir · 本地運行版")
