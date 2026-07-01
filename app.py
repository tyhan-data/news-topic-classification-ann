"""
NewsCortex — AI News Desk
==========================
A Streamlit app that classifies a raw news headline / article snippet into
one of four editorial desks: World, Sports, Business, or Sci/Tech.

PIPELINE
--------
Raw text  --->  TF-IDF vectorizer (10,000 terms, uni+bigrams)  --->
Dense neural network (6 hidden blocks, BatchNorm + Dropout)  --->
Softmax over 4 classes

This file is organised top-to-bottom in the order the page actually renders,
and every section is commented so a non-ML reader (e.g. a recruiter glancing
at the code) can follow what each part is doing.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

# ----------------------------------------------------------------------------
# 1. PAGE CONFIG  (must be the first Streamlit call in the script)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewsCortex — AI News Desk",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "ag_news_classifier.keras"
VECTORIZER_PATH = APP_DIR / "tfidf_vectorizer.pkl"

# ----------------------------------------------------------------------------
# 2. STATIC CONTENT  (class metadata, sample headlines, demo text)
#    Keeping all the "content" in one place makes the app easy to re-skin
#    or re-purpose for a different label set later.
# ----------------------------------------------------------------------------
CLASSES = [
    {
        "name": "World",
        "color": "#3FA7A2",
        "desk": "FOREIGN DESK",
        "blurb": "Geopolitics, diplomacy, conflict & international affairs.",
    },
    {
        "name": "Sports",
        "color": "#E15554",
        "desk": "SPORTS DESK",
        "blurb": "Match results, transfers, records & competitions.",
    },
    {
        "name": "Business",
        "color": "#4CAF6D",
        "desk": "MARKETS DESK",
        "blurb": "Markets, earnings, trade & corporate news.",
    },
    {
        "name": "Sci/Tech",
        "color": "#8C7AE6",
        "desk": "SCI-TECH DESK",
        "blurb": "Research, computing, gadgets & engineering.",
    },
]

# Short, original wire-style headlines used purely for the decorative ticker.
TICKER_HEADLINES = [
    ("World", "Leaders from twelve nations sign new regional security pact"),
    ("Sports", "Underdog club stuns league leaders with late extra-time goal"),
    ("Business", "Central bank holds rates steady, citing cooling inflation"),
    ("Sci/Tech", "Researchers unveil battery design that charges in minutes"),
    ("World", "Peace talks resume after months of stalled negotiations"),
    ("Sports", "Veteran sprinter breaks two-decade-old national record"),
    ("Business", "Retailer reports record quarterly profit on holiday sales"),
    ("Sci/Tech", "New space telescope returns clearest images yet of galaxy"),
]

# One-click demo snippets — original text, hand-checked to land confidently
# in the expected class so the demo always feels convincing.
DEMO_SAMPLES = {
    "World": "The United Nations Security Council met today to discuss the "
             "ongoing conflict and diplomatic relations between the two nations.",
    "Sports": "The home team clinched the title after a last-minute goal sent "
              "fans into a frenzy during the final match of the season.",
    "Business": "Shares of the company surged after it posted stronger than "
                "expected quarterly earnings and raised its full-year revenue forecast.",
    "Sci/Tech": "Engineers unveiled a new processor chip that promises significantly "
                "faster computing performance while consuming less power.",
}

CLASS_LOOKUP = {c["name"]: c for c in CLASSES}

# ----------------------------------------------------------------------------
# 3. THEME / CSS
#    Streamlit's default look is intentionally overridden here so the app
#    reads like a "wire service" news desk rather than a generic ML demo.
#    Everything is plain CSS injected once at the top of the page.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,700;0,800;1,500&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

    :root{
        --ink:#14171C;
        --ink-light:#1E2229;
        --paper:#F1EFE9;
        --slate:#9AA0AA;
        --amber:#E8A33D;
        --rule:rgba(241,239,233,0.12);
    }

    /* ---- base page ---- */
    .stApp { background-color: var(--ink); }
    [data-testid="stAppViewContainer"], .stApp, body { color: var(--paper); }
    [data-testid="stSidebar"] { background-color: var(--ink-light); border-right: 1px solid var(--rule); }
    [data-testid="stHeader"] { background: transparent; }

    /* ---- typography ---- */
    h1, h2, h3 { font-family: 'Spectral', serif; }
    p, span, div, label { font-family: 'Inter', sans-serif; }
    code, .mono { font-family: 'JetBrains Mono', monospace; }

    /* ---- masthead ---- */
    .eyebrow{
        font-family:'JetBrains Mono', monospace;
        letter-spacing:.18em; text-transform:uppercase;
        font-size:0.72rem; color: var(--amber); font-weight:700;
        margin-bottom: 0.35rem;
    }
    .masthead-title{
        font-family:'Spectral', serif; font-weight:800; font-style:italic;
        font-size: clamp(2.2rem, 5vw, 3.6rem);
        color: var(--paper); line-height:1.0; margin:0;
        border-bottom: 3px solid var(--paper);
        display:inline-block; padding-bottom: 0.15rem;
    }
    .masthead-tagline{
        color: var(--slate); font-size: 1.0rem; margin-top: 0.6rem; max-width: 640px;
    }
    .status-line{
        font-family:'JetBrains Mono', monospace; font-size:0.74rem; color: var(--slate);
        margin-top: 0.9rem; padding-top: 0.6rem; border-top: 1px solid var(--rule);
        letter-spacing: 0.03em;
    }
    .status-line b{ color: var(--paper); }

    /* ---- wire ticker (signature element) ---- */
    .ticker-wrap{
        overflow:hidden; white-space:nowrap; background: var(--ink-light);
        border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
        margin: 1.1rem 0 1.6rem 0; padding: 0.5rem 0;
    }
    .ticker-track{
        display:inline-block; white-space:nowrap;
        animation: scroll-left 38s linear infinite;
        font-family:'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--slate);
    }
    .ticker-wrap:hover .ticker-track{ animation-play-state: paused; }
    .ticker-track span.tag{ font-weight:700; letter-spacing:.05em; }
    .ticker-track span.sep{ color: var(--rule); margin: 0 0.9rem; }
    @keyframes scroll-left{
        0%{ transform: translateX(0); }
        100%{ transform: translateX(-50%); }
    }
    @media (prefers-reduced-motion: reduce){
        .ticker-track{ animation: none; }
    }

    /* ---- cards ---- */
    .panel{
        background: var(--ink-light); border: 1px solid var(--rule);
        border-radius: 6px; padding: 1.4rem 1.4rem 1.2rem 1.4rem;
        height: 100%;
    }
    .panel-title{
        font-family:'JetBrains Mono', monospace; letter-spacing:.12em;
        text-transform:uppercase; font-size:0.78rem; color: var(--amber);
        font-weight:700; margin-bottom: 0.9rem;
    }

    /* ---- idle / placeholder state ---- */
    .idle-box{
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        height: 230px; color: var(--slate); text-align:center;
        font-family:'JetBrains Mono', monospace; font-size: 0.85rem;
        border: 1px dashed var(--rule); border-radius: 6px;
    }

    /* ---- predicted desk badge ---- */
    .desk-badge{
        display:inline-flex; align-items:center; gap:0.5rem;
        padding: 0.4rem 0.9rem; border-radius: 999px;
        font-family:'JetBrains Mono', monospace; font-weight:700; font-size: 0.95rem;
        letter-spacing: 0.04em;
    }
    .desk-dot{ width:10px; height:10px; border-radius:50%; display:inline-block; }

    /* ---- confidence bars ---- */
    .bar-row{ margin: 0.55rem 0; }
    .bar-label{
        display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace;
        font-size: 0.8rem; color: var(--paper); margin-bottom: 0.25rem;
    }
    .bar-track{ background: rgba(255,255,255,0.06); border-radius: 4px; height: 9px; overflow:hidden; }
    .bar-fill{ height: 100%; border-radius: 4px; }

    /* ---- architecture table ---- */
    table.arch-table{ width:100%; border-collapse: collapse; font-family:'JetBrains Mono', monospace; font-size:0.78rem; }
    table.arch-table th{ text-align:left; color: var(--amber); border-bottom: 1px solid var(--rule); padding: 0.4rem 0.5rem; }
    table.arch-table td{ color: var(--paper); border-bottom: 1px solid var(--rule); padding: 0.35rem 0.5rem; }
    table.arch-table tr:last-child td{ border-bottom:none; }

    /* ---- footer ---- */
    .chip{
        display:inline-block; font-family:'JetBrains Mono', monospace; font-size:0.72rem;
        border:1px solid var(--rule); color: var(--slate); padding: 0.2rem 0.6rem;
        border-radius: 999px; margin: 0.2rem 0.3rem 0.2rem 0;
    }

    /* ---- inputs ---- */
    [data-testid="stTextArea"] textarea{
        background: var(--ink); color: var(--paper); border: 1px solid var(--rule);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stButton"] button{
        font-family:'JetBrains Mono', monospace; font-weight:600; letter-spacing:0.03em;
    }

    /* ---- responsive tweaks ---- */
    @media (max-width: 640px){
        .masthead-title{ font-size: 2.1rem; }
        .status-line{ font-size: 0.66rem; }
        .ticker-track{ font-size: 0.74rem; }
        .panel{ padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 4. MODEL LOADING  (cached so the ~31MB model & vectorizer load only once
#    per server process, not on every user interaction / rerun)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str, model, vectorizer) -> np.ndarray:
    """Turn raw text into class probabilities. Returns a length-4 array."""
    features = vectorizer.transform([text]).toarray()
    probabilities = model.predict(features, verbose=0)[0]
    return probabilities


# ----------------------------------------------------------------------------
# 5. SIDEBAR  — project context for a technical / recruiter audience
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗞️ NewsCortex")
    st.markdown(
        "An end-to-end NLP project: a **TF-IDF + deep neural network** "
        "pipeline that reads a news snippet and assigns it to the correct desk."
    )

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. Raw text\n"
        "2. TF-IDF vectorizer → 10,000 features (uni + bigrams)\n"
        "3. Dense neural network → 6 hidden blocks (Dense → BatchNorm → ReLU → Dropout)\n"
        "4. Softmax → 4-class probability distribution"
    )

    st.markdown("---")
    st.markdown("**Desks it sorts into**")
    for c in CLASSES:
        st.markdown(
            f'<span class="desk-dot" style="background:{c["color"]}; '
            f'display:inline-block; width:9px; height:9px; border-radius:50%; '
            f'margin-right:6px;"></span>'
            f'<span class="mono">{c["name"]}</span> — {c["blurb"]}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Tech stack**")
    st.markdown(
        '<span class="chip">Python</span><span class="chip">TensorFlow / Keras</span>'
        '<span class="chip">scikit-learn</span><span class="chip">Streamlit</span>'
        '<span class="chip">NumPy</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Built as a portfolio project demonstrating a full NLP "
               "classification pipeline, from text vectorization to a "
               "deployed, interactive interface.")

# ----------------------------------------------------------------------------
# 6. MASTHEAD
# ----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">AUTOMATED NEWS DESK · EST. 2026</div>', unsafe_allow_html=True)
st.markdown('<h1 class="masthead-title">NewsCortex</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="masthead-tagline">Feed it a raw headline or a few lines of an '
    'article — it reads the text and instantly files it under the right desk: '
    'World, Sports, Business, or Sci/Tech.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="status-line">MODEL: <b>6-block Dense NN</b> &nbsp;·&nbsp; '
    'FEATURES: <b>10,000 TF-IDF terms</b> &nbsp;·&nbsp; CLASSES: <b>4</b> '
    '&nbsp;·&nbsp; FRAMEWORK: <b>TensorFlow / Keras</b></div>',
    unsafe_allow_html=True,
)

# ---- wire ticker: scrolling strip of sample headlines (purely decorative) --
ticker_items = TICKER_HEADLINES * 2  # duplicated so the loop has no visible seam
ticker_html = ""
for cat, headline in ticker_items:
    color = CLASS_LOOKUP[cat]["color"]
    ticker_html += (
        f'<span class="tag" style="color:{color};">{cat.upper()}</span> '
        f'<span>{headline}</span><span class="sep">//</span>'
    )
st.markdown(
    f'<div class="ticker-wrap"><div class="ticker-track">{ticker_html}</div></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 7. LOAD MODEL  (and fail gracefully with a clear message if files are missing)
# ----------------------------------------------------------------------------
model, vectorizer = load_pipeline()
if model is None:
    st.error(
        "Model files not found. Place **ag_news_classifier.keras** and "
        "**tfidf_vectorizer.pkl** in the same folder as `app.py`, then rerun."
    )
    st.stop()

# Keep the latest prediction in session_state so it survives reruns
# triggered by widgets other than the classify button (e.g. resizing).
if "probs" not in st.session_state:
    st.session_state.probs = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ----------------------------------------------------------------------------
# 8. MAIN LAYOUT — input panel (left) + results panel (right)
# ----------------------------------------------------------------------------
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📝 Submit a story</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "Headline or article snippet",
        value=st.session_state.input_text,
        height=160,
        placeholder="e.g. \"The central bank raised interest rates again today, "
                    "citing persistent inflation pressure across the economy.\"",
        label_visibility="collapsed",
    )

    st.caption("Or try a one-click sample:")
    sample_cols = st.columns(4)
    for i, (label, sample_text) in enumerate(DEMO_SAMPLES.items()):
        with sample_cols[i]:
            if st.button(label, use_container_width=True, key=f"sample_{label}"):
                st.session_state.input_text = sample_text
                st.rerun()

    classify_clicked = st.button("Classify story →", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Run the model only when the button is clicked, on whatever text is in the box.
if classify_clicked:
    if not text_input.strip():
        st.toast("Type or paste some text first.", icon="⚠️")
    else:
        with st.spinner("Wiring text through the network..."):
            time.sleep(0.3)  # tiny pause so the spinner is perceptible — pure UX polish
            st.session_state.probs = predict(text_input, model, vectorizer)
            st.session_state.input_text = text_input

with col_result:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📡 Desk assignment</div>', unsafe_allow_html=True)

    if st.session_state.probs is None:
        st.markdown(
            '<div class="idle-box">— AWAITING INPUT —<br>'
            'Submit a story to see the model\'s call.</div>',
            unsafe_allow_html=True,
        )
    else:
        probs = st.session_state.probs
        order = np.argsort(probs)[::-1]          # rank classes by confidence
        top_idx = order[0]
        top_class = CLASSES[top_idx]

        st.markdown(
            f'<span class="desk-badge" style="background:{top_class["color"]}22; '
            f'color:{top_class["color"]}; border:1px solid {top_class["color"]}66;">'
            f'<span class="desk-dot" style="background:{top_class["color"]};"></span>'
            f'{top_class["desk"]} — {top_class["name"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="color:var(--slate); margin-top:0.7rem; font-size:0.88rem;">'
            f'{top_class["blurb"]}</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        for idx in order:
            c = CLASSES[idx]
            pct = probs[idx] * 100
            st.markdown(
                f"""
                <div class="bar-row">
                    <div class="bar-label"><span>{c['name']}</span><span>{pct:.1f}%</span></div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{pct:.1f}%; background:{c['color']};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 9. "UNDER THE HOOD" — architecture detail for technical reviewers
# ----------------------------------------------------------------------------
with st.expander("🔧 Under the hood — inspect the actual model & vectorizer"):
    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("**TF-IDF Vectorizer**")
        vec_rows = [
            ("Vocabulary size", f"{len(vectorizer.vocabulary_):,} terms"),
            ("N-gram range", str(vectorizer.ngram_range)),
            ("Stop words", str(vectorizer.stop_words)),
            ("Sublinear TF scaling", str(vectorizer.sublinear_tf)),
            ("Normalization", str(vectorizer.norm)),
        ]
        rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in vec_rows)
        st.markdown(
            f'<table class="arch-table"><tr><th>Setting</th><th>Value</th></tr>{rows_html}</table>',
            unsafe_allow_html=True,
        )

    with tech_col2:
        st.markdown("**Network layers** (read live from the loaded model)")
        layer_rows = ""
        for layer in model.layers:
            try:
                shape = layer.output.shape
            except Exception:
                shape = "—"
            params = layer.count_params()
            layer_rows += (
                f"<tr><td>{layer.__class__.__name__}</td>"
                f"<td>{shape}</td><td>{params:,}</td></tr>"
            )
        st.markdown(
            f'<table class="arch-table"><tr><th>Layer</th><th>Output shape</th>'
            f'<th>Params</th></tr>{layer_rows}</table>',
            unsafe_allow_html=True,
        )
        st.caption(f"Total parameters: {model.count_params():,}")

# ----------------------------------------------------------------------------
# 10. FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="status-line" style="margin-top:2rem;">'
    'NewsCortex is a portfolio project — predictions are illustrative, not '
    'editorial fact-checking.</div>',
    unsafe_allow_html=True,
)
