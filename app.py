import html
import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.model import predict_answer
from src.preprocessing import validate_qa_inputs


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Transformer-QA | Extractive Question Answering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results" / "model_comparison_results.csv"

MODEL_CONFIGS = {
    "RoBERTa": "deepset/roberta-base-squad2",
    "DistilBERT": "distilbert/distilbert-base-uncased-distilled-squad",
    "BERT-Cased": "deepset/bert-base-cased-squad2",
}

MODEL_LABELS = {
    "RoBERTa": "👑 RoBERTa — Best Accuracy",
    "DistilBERT": "⚡ DistilBERT — Fastest",
    "BERT-Cased": "🔷 BERT-Cased — Comparison",
}

DEFAULT_CONTEXT = (
    "Bangladesh is a country in South Asia. "
    "Its capital is Dhaka. "
    "It is known for its rich culture and history."
)

DEFAULT_QUESTION = "What is the capital of Bangladesh?"

REQUIRED_RESULT_COLUMNS = (
    "Model",
    "Exact Match (%)",
    "F1 Score (%)",
    "Avg Time/Question (s)",
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>
:root {
    --navy: #081d50;
    --purple: #6134dc;
    --text: #17244c;
    --muted: #74819c;
}

.stApp {
    background: #f7f9ff;
    color: var(--text);
}

.block-container {
    max-width: 1150px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #17244c;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7f8ff, #eef1ff);
    border-right: 1px solid #e1e7f4;
}

.sidebar-brand {
    padding: 12px 0 16px;
}

.sidebar-brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #12275b;
}

.sidebar-brand-subtitle {
    font-size: 0.8rem;
    color: #7885a0;
    margin-top: 5px;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.4rem;
}

[data-testid="stSidebar"] label[data-baseweb="radio"] {
    padding: 10px 12px;
    border-radius: 10px;
    transition: background 0.2s;
}

[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background: #e9ebff;
}

[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
    background: #e1e4ff;
}

.feature-panel {
    background: #e9eaff;
    border: 1px solid #e0e2fa;
    border-radius: 14px;
    padding: 17px;
    margin-top: 25px;
}

.feature-heading {
    font-weight: 800;
    color: #2c2d7b;
    font-size: 0.96rem;
    margin-bottom: 14px;
}

.feature-item {
    font-size: 0.84rem;
    color: #484a83;
    margin: 10px 0;
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    min-height: 220px;
    border-radius: 18px;
    padding: 32px 40px;
    margin-bottom: 25px;
    background:
        radial-gradient(
            ellipse at 86% 40%,
            rgba(117, 66, 245, 0.48),
            transparent 35%
        ),
        radial-gradient(
            ellipse at 15% 100%,
            rgba(24, 104, 200, 0.4),
            transparent 45%
        ),
        linear-gradient(
            120deg,
            #06183e 0%,
            #082659 65%,
            #101d5c 100%
        );
    box-shadow: 0 10px 26px rgba(13, 36, 88, 0.12);
}

.hero-brand {
    color: #dbe8ff;
    font-size: 0.86rem;
    font-weight: 650;
    margin-bottom: 17px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(2rem, 3.7vw, 3rem);
    font-weight: 850;
    line-height: 1.1;
    margin: 0 0 13px;
    letter-spacing: -0.04em;
}

.hero h1 span {
    color: #b08cff;
}

.hero-description {
    color: #e4ecff;
    font-size: 0.96rem;
    line-height: 1.6;
    max-width: 480px;
    margin-bottom: 17px;
}

.hero-pill {
    display: inline-block;
    padding: 7px 14px;
    background: rgba(41, 127, 221, 0.37);
    border: 1px solid rgba(154, 192, 255, 0.25);
    color: #f2f6ff;
    border-radius: 25px;
    font-size: 0.77rem;
    font-weight: 700;
}

/* CHIP ILLUSTRATION */

.hero-art {
    position: absolute;
    top: 33px;
    right: 75px;
    width: 165px;
    height: 165px;
    transform: rotate(-24deg);
}

.chip-layer {
    position: absolute;
    left: 23px;
    width: 115px;
    height: 115px;
    border: 2px solid #629dff;
    border-radius: 17px;
    transform: rotate(45deg) scaleY(0.65);
    box-shadow: 0 0 22px rgba(100, 93, 255, 0.45);
}

.chip-layer.bottom {
    top: 38px;
    background: #173da7;
    opacity: 0.65;
}

.chip-layer.middle {
    top: 23px;
    background: #5430c5;
    opacity: 0.9;
}

.chip-layer.top {
    top: 5px;
    background: linear-gradient(140deg, #164ce4, #7628d9);
    display: flex;
    align-items: center;
    justify-content: center;
}

.chip-core {
    width: 55px;
    height: 55px;
    border: 3px solid #c9d7ff;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 2rem;
    transform: scaleY(1.5);
}

/* MAIN SECTION */

.section-heading {
    font-size: 1.5rem;
    font-weight: 800;
    color: #142451;
    margin: 7px 0 5px;
}

.section-caption {
    font-size: 0.88rem;
    color: #7783a0;
    margin-bottom: 18px;
}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    border: 1px solid #d5ddef;
    border-radius: 9px;
    background: white;
    color: #1b294c;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #3936c9, #6028db);
    color: white;
    border: none;
    min-height: 46px;
    border-radius: 9px;
    font-weight: 750;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #2b2ba9, #4f1fc2);
    color: white;
}

/* ANSWER CARD */

.answer-card {
    background: linear-gradient(125deg, #e2ffed, #d7f8e7);
    border: 1px solid #bcebd0;
    border-radius: 13px;
    padding: 18px 20px;
    min-height: 115px;
}

.answer-label {
    font-size: 0.93rem;
    font-weight: 800;
    color: #1a7647;
    margin-bottom: 11px;
}

.answer-value {
    font-size: 1.8rem;
    font-weight: 850;
    color: #123d29;
    overflow-wrap: anywhere;
}

/* MODEL INFO */

.model-info-card {
    background: #edf2ff;
    border: 1px solid #dce5fc;
    border-radius: 13px;
    padding: 17px 20px;
    min-height: 115px;
}

.model-info-row {
    font-size: 0.85rem;
    color: #34466c;
    margin: 8px 0;
}

.model-info-row strong {
    color: #1b2c5a;
}

/* COMPARISON TABLE */

.comparison-title {
    color: #172857;
    font-size: 1.2rem;
    font-weight: 800;
    margin: 10px 0 6px;
}

.comparison-caption {
    color: #7b87a2;
    font-size: 0.82rem;
    margin-bottom: 14px;
}

.table-wrap {
    border: 1px solid #e1e7f2;
    border-radius: 11px;
    overflow-x: auto;
    background: white;
}

.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    color: #25345b;
}

.comparison-table th {
    background: #f0f3fa;
    color: #23345b;
    text-align: left;
    padding: 12px;
    font-weight: 800;
    white-space: nowrap;
}

.comparison-table td {
    padding: 11px 12px;
    border-top: 1px solid #e8edf5;
    white-space: nowrap;
}

.comparison-table tr.best-model {
    background: #e7eaff;
    color: #24257c;
    font-weight: 800;
}

.comparison-table tbody tr:hover {
    background: #f1f5ff;
}

@media (max-width: 900px) {
    .hero-art {
        right: 12px;
        transform: rotate(-24deg) scale(0.7);
        opacity: 0.55;
    }

    .hero-description {
        max-width: 330px;
    }
}

@media (max-width: 650px) {
    .hero-art {
        display: none;
    }

    .hero {
        min-height: auto;
        padding: 27px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 3. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_qa_model(model_name):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# =========================================================
# 4. LOAD AND VALIDATE COMPARISON DATA
# =========================================================

@st.cache_data(ttl=10)
def load_comparison_data():
    if not RESULTS_FILE.is_file():
        return (
            None,
            "Model comparison results not found. "
            "Run python -m src.model_comparison first.",
        )

    try:
        data = pd.read_csv(RESULTS_FILE)

    except (
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeError,
    ) as error:
        return None, f"Could not read comparison results: {error}"

    missing = [
        column
        for column in REQUIRED_RESULT_COLUMNS
        if column not in data.columns
    ]

    if missing:
        return (
            None,
            "Comparison CSV is missing columns: "
            + ", ".join(missing),
        )

    if data.empty:
        return None, "Comparison CSV has no model results."

    data = data.copy()

    if (
        data["Model"].isna().any()
        or data["Model"].astype(str).str.strip().eq("").any()
    ):
        return None, "Comparison CSV contains an empty model name."

    for column in REQUIRED_RESULT_COLUMNS[1:]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if not data[column].notna().all():
            return (
                None,
                f"Comparison CSV contains invalid values in {column}.",
            )

    return data, None


# =========================================================
# 5. HERO BANNER
# =========================================================

def render_hero():
    hero_html = """
<div class="hero">
    <div class="hero-brand">▣ Transformer-QA</div>
    <h1>Transformer-<span>QA</span></h1>
    <div class="hero-description">
        Extractive Question Answering with
        Pre-trained Transformer Models
    </div>
    <div class="hero-pill">
        BERT · RoBERTa · DistilBERT
    </div>
    <div class="hero-art">
        <div class="chip-layer bottom"></div>
        <div class="chip-layer middle"></div>
        <div class="chip-layer top">
            <div class="chip-core">✦</div>
        </div>
    </div>
</div>
"""

    st.markdown(hero_html, unsafe_allow_html=True)


# =========================================================
# 6. SIDEBAR
# =========================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-brand">
    <div class="sidebar-brand-title">▣ Transformer-QA</div>
    <div class="sidebar-brand-subtitle">
        Transformer-based Extractive QA
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            [
                "💬 Ask a Question",
                "📊 Model Comparison",
                "🧊 About Models",
                "📄 Project Info",
            ],
            label_visibility="collapsed",
        )

        st.markdown(
            """
<div class="feature-panel">
    <div class="feature-heading">✦ Key Features</div>
    <div class="feature-item">⚡ Three Transformer Models</div>
    <div class="feature-item">🎯 Extractive Question Answering</div>
    <div class="feature-item">📊 EM &amp; F1 Evaluation</div>
    <div class="feature-item">🚀 Interactive Web App</div>
</div>
""",
            unsafe_allow_html=True,
        )

    return page


# =========================================================
# 7. MODEL COMPARISON TABLE
# =========================================================

def render_comparison_table():
    data, error = load_comparison_data()

    if error is not None:
        st.warning(error)
        return

    best_model = data.loc[
        data["F1 Score (%)"].idxmax(),
        "Model",
    ]

    rows = []

    for _, item in data.iterrows():
        model_name = str(item["Model"])

        row_class = (
            ' class="best-model"'
            if model_name == best_model
            else ""
        )

        rows.append(
            f'<tr{row_class}>'
            f'<td>{html.escape(model_name)}</td>'
            f'<td>{float(item["Exact Match (%)"]):.2f}%</td>'
            f'<td>{float(item["F1 Score (%)"]):.2f}%</td>'
            f'<td>{float(item["Avg Time/Question (s)"]):.3f}s</td>'
            '</tr>'
        )

    table_html = (
        '<div class="table-wrap">'
        '<table class="comparison-table">'
        '<thead><tr>'
        '<th>Model</th>'
        '<th>Exact Match</th>'
        '<th>F1 Score</th>'
        '<th>Avg. Time / Q</th>'
        '</tr></thead>'
        '<tbody>'
        + ''.join(rows)
        + '</tbody></table></div>'
    )

    st.markdown(
        table_html,
        unsafe_allow_html=True,
    )


# =========================================================
# 8. QUESTION ANSWERING PAGE
# =========================================================

def render_qa_page():
    st.markdown(
        '<div class="section-heading">💬 Ask a Question</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Provide a context and a question. '
        'Select a model to extract the answer.'
        '</div>',
        unsafe_allow_html=True,
    )

    model_label = st.selectbox(
        "Select Model",
        options=list(MODEL_CONFIGS.keys()),
        index=0,
        format_func=lambda name: MODEL_LABELS[name],
    )

    context = st.text_area(
        "Context",
        value=DEFAULT_CONTEXT,
        height=135,
        max_chars=20000,
    )

    question = st.text_input(
        "Question",
        value=DEFAULT_QUESTION,
    )

    submitted = st.button(
        "▶ Get Answer",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        # Remove previous prediction before processing a new request.
        st.session_state.pop("qa_result", None)

        if not context.strip():
            st.warning("Please enter a context.")

        elif not question.strip():
            st.warning("Please enter a question.")

        else:
            checkpoint = MODEL_CONFIGS[model_label]

            try:
                with st.spinner(f"Loading {model_label}..."):
                    tokenizer, model, device = load_qa_model(
                        checkpoint
                    )

                # Validate inputs before running model inference.
                validate_qa_inputs(
                    question,
                    context,
                    tokenizer,
                )

                with st.spinner("Finding the answer..."):
                    start_time = time.perf_counter()

                    answer, span_score = predict_answer(
                        question,
                        context,
                        model,
                        tokenizer,
                        device,
                    )

                    elapsed_time = (
                        time.perf_counter() - start_time
                    )

                st.session_state["qa_result"] = {
                    "answer": answer,
                    "model": model_label,
                    "time": elapsed_time,
                    "score": span_score,
                    "context": context,
                    "question": question,
                }

            except ValueError as error:
                st.warning(str(error))

            except Exception:
                logging.exception(
                    "Transformer-QA prediction failed"
                )

                st.error(
                    "Prediction failed due to a model or "
                    "application error. Please check the "
                    "terminal logs and try again."
                )

    result = st.session_state.get("qa_result")

    if result is not None:
        is_current = (
            result["context"] == context
            and result["question"] == question
            and result["model"] == model_label
        )

        if is_current:
            left, right = st.columns(
                [1.1, 1],
                gap="medium",
            )

            with left:
                safe_answer = html.escape(
                    str(
                        result["answer"]
                        or "No answer extracted"
                    )
                )

                answer_html = (
                    '<div class="answer-card">'
                    '<div class="answer-label">'
                    '✓ Extracted Answer'
                    '</div>'
                    f'<div class="answer-value">'
                    f'{safe_answer}'
                    '</div>'
                    '</div>'
                )

                st.markdown(
                    answer_html,
                    unsafe_allow_html=True,
                )

            with right:
                safe_model = html.escape(
                    result["model"]
                )

                info_html = (
                    '<div class="model-info-card">'
                    '<div class="model-info-row">'
                    f'◈ <strong>Model:</strong> {safe_model}'
                    '</div>'
                    '<div class="model-info-row">'
                    '◷ <strong>Response Time:</strong> '
                    f'{result["time"]:.3f} seconds'
                    '</div>'
                    '<div class="model-info-row">'
                    '◉ <strong>Raw Span Score:</strong> '
                    f'{result["score"]:.2f}'
                    '</div>'
                    '</div>'
                )

                st.markdown(
                    info_html,
                    unsafe_allow_html=True,
                )

    st.divider()

    st.markdown(
        '<div class="comparison-title">'
        '📊 Model Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="comparison-caption">'
        'Evaluation on 200 SQuAD validation questions'
        '</div>',
        unsafe_allow_html=True,
    )

    render_comparison_table()


# =========================================================
# 9. MODEL COMPARISON PAGE
# =========================================================

def render_comparison_page():
    st.markdown(
        '<div class="section-heading">'
        '📊 Model Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Performance comparison of three pretrained '
        'Transformer checkpoints on 200 SQuAD questions.'
        '</div>',
        unsafe_allow_html=True,
    )

    render_comparison_table()

    data, error = load_comparison_data()

    if error is None:
        st.subheader("Performance Visualization")

        chart_data = data.set_index("Model")[
            ["Exact Match (%)", "F1 Score (%)"]
        ]

        st.bar_chart(chart_data)

        st.caption(
            "The selected checkpoints were fine-tuned "
            "on different SQuAD versions. "
            "This compares checkpoints, not architectures alone."
        )


# =========================================================
# 10. ABOUT MODELS PAGE
# =========================================================

def render_models_page():
    st.markdown(
        '<div class="section-heading">'
        '🧠 About the Models'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("RoBERTa")

        st.write(
            "A robustly optimized BERT-based Transformer. "
            "It achieved the highest EM and F1 scores "
            "in our evaluation."
        )

    with st.container(border=True):
        st.subheader("DistilBERT")

        st.write(
            "A compact Transformer model designed for "
            "faster inference. It was the fastest model "
            "in our comparison."
        )

    with st.container(border=True):
        st.subheader("BERT-Cased")

        st.write(
            "A BERT checkpoint that preserves capitalization "
            "and was fine-tuned on SQuAD2."
        )


# =========================================================
# 11. PROJECT INFORMATION PAGE
# =========================================================

def render_project_page():
    st.markdown(
        '<div class="section-heading">'
        '📄 Project Information'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Transformer-QA is an extractive question-answering "
        "application built using pretrained Transformer models."
    )

    st.markdown(
        """
**Dataset:** SQuAD v1.1

**Models:** RoBERTa, DistilBERT, BERT-Cased

**Evaluation:** Exact Match and Token-level F1

**Technologies:** Python, PyTorch, Hugging Face,
Pandas, Streamlit

**Task:** Extract answers directly from user-provided context.
"""
    )

    st.info(
        "The application uses extractive QA. "
        "Answers are selected from the supplied context, "
        "not generated as new text."
    )


# =========================================================
# 12. MAIN APPLICATION
# =========================================================

def main():
    page = render_sidebar()

    render_hero()

    if page == "💬 Ask a Question":
        render_qa_page()

    elif page == "📊 Model Comparison":
        render_comparison_page()

    elif page == "🧊 About Models":
        render_models_page()

    elif page == "📄 Project Info":
        render_project_page()


if __name__ == "__main__":
    main()