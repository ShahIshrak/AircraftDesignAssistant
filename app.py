from __future__ import annotations

# Force Hugging Face out of offline mode and enable online downloads
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import nltk
import streamlit as st

# Download the required NLTK resource silently before importing other modules
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# =============================================================================
# app.py — Aircraft Design Assistant  (Streamlit front-end)
# =============================================================================
#
# All design requirements are entered as free-form natural language in a
# single text box — exactly how AircraftDesignAssistant.start() expects them.
# No dropdowns, no number inputs, no selectboxes.
#
# The assistant's LLM (Stage 0) extracts every structured parameter from the
# text, so users just describe what they want in plain English.
#
# Pipeline (all inside AircraftDesignAssistant.start()):
#     Stage 0  — LLM extracts structured spec from free text
#     Stage 1  — Literature / reference design search
#     Stage 1b — LLM suggests airfoil candidates (wing + tail)
#     Stage 2a — AirfoilAnalysisAgent: XFOIL / panel analysis, wing candidates
#     Stage 2b — AirfoilAnalysisAgent: XFOIL / panel analysis, tail candidates
#     Stage 3  — Preliminary sizing (wing, tail, fuselage, propulsion)
#     Stage 3b — Engineering constraint search
#     Stage 4  — LLM synthesis → design assessment report
#
# Run:  streamlit run app.py
# =============================================================================

import io
import queue
import sys
import threading
import time
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft Design Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark engineering aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

.stApp { background: #080c12; color: #c9d4e0; }

h1 {
    font-family: 'Barlow', sans-serif; font-weight: 700;
    letter-spacing: .04em; color: #e4ebf5;
    border-bottom: 2px solid #1d4ed8;
    padding-bottom: .3rem; margin-bottom: .8rem;
}
h2 { font-family: 'Barlow', sans-serif; font-weight: 600; color: #93c5fd; }
h3 { font-family: 'Barlow', sans-serif; font-weight: 600; color: #7dd3fc; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0c1119; border-right: 1px solid #1a2236; }
section[data-testid="stSidebar"] .stMarkdown p { color: #9ca3af; font-size: .85rem; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #0f1520; border: 1px solid #1a2236;
    border-left: 3px solid #1d4ed8; border-radius: 6px; padding: .6rem .9rem;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important; font-size: .68rem !important;
    text-transform: uppercase; letter-spacing: .07em;
}
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.15rem !important;
}

/* The main text input */
.stTextArea textarea {
    background: #0d1421 !important;
    border: 1px solid #1e3a6e !important;
    border-radius: 8px !important;
    color: #e2e8f4 !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: .95rem !important;
    line-height: 1.65 !important;
    caret-color: #60a5fa;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,.25) !important;
    outline: none !important;
}
label { color: #6b7280 !important; font-size: .8rem !important; letter-spacing: .04em; }

/* Run button */
.stButton > button {
    background: #1d4ed8; color: #fff; border: none; border-radius: 6px;
    font-family: 'Barlow', sans-serif; font-weight: 700;
    font-size: 1rem; letter-spacing: .06em;
    padding: .65rem 2rem; width: 100%;
    transition: background .18s, transform .1s;
}
.stButton > button:hover  { background: #2563eb; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #1d4ed8, #38bdf8);
    border-radius: 4px;
}

/* Alerts */
.stSuccess { background: #031a0e; border: 1px solid #166534; color: #86efac; border-radius: 5px; }
.stInfo    { background: #091433; border: 1px solid #1e40af; color: #93c5fd; border-radius: 5px; }
.stWarning { background: #1a0e00; border: 1px solid #92400e; color: #fcd34d; border-radius: 5px; }
.stError   { background: #1a0509; border: 1px solid #991b1b; color: #fca5a5; border-radius: 5px; }

/* Expander */
.streamlit-expanderHeader {
    background: #0f1520 !important; color: #6b7280 !important;
    font-weight: 600; border: 1px solid #1a2236; border-radius: 5px;
}

/* Console */
.console-wrap {
    background: #040810; border: 1px solid #1a2236; border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .71rem; line-height: 1.55; color: #22c55e;
    padding: .9rem 1.1rem; height: 360px;
    overflow-y: auto; white-space: pre-wrap; word-break: break-all;
}

/* Report block */
.result-block {
    background: #0a0f1a; border: 1px solid #1a2236; border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .74rem; line-height: 1.65; color: #cbd5e1;
    padding: 1rem 1.3rem; white-space: pre-wrap;
    overflow-x: auto; max-height: 640px; overflow-y: auto;
}

/* Prompt chip styling */
.prompt-chip {
    display: inline-block;
    background: #0f1a2e; border: 1px solid #1e3a6e;
    border-radius: 20px; padding: .25rem .75rem;
    font-size: .78rem; color: #93c5fd;
    margin: .2rem .2rem; cursor: pointer;
    transition: background .15s;
}
.prompt-chip:hover { background: #1e3a6e; }

.section-label {
    color: #374151; font-size: .68rem; text-transform: uppercase;
    letter-spacing: .12em; font-weight: 700; margin: .8rem 0 .2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe stdout tee → queue
# ─────────────────────────────────────────────────────────────────────────────

class _QueueWriter(io.TextIOBase):
    def __init__(self, q: "queue.Queue[str | None]", original: io.TextIOBase):
        self._q        = q
        self._original = original

    def write(self, text: str) -> int:
        if text:
            self._q.put(text)
            try:
                self._original.write(text)
                self._original.flush()
            except Exception:
                pass
        return len(text)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass


def _install_capture(q: "queue.Queue[str | None]"):
    sys.stdout = _QueueWriter(q, sys.__stdout__)

def _restore_stdout():
    sys.stdout = sys.__stdout__


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "console_lines"  : [],
    "design_result"  : None,
    "design_running" : False,
    "design_done"    : False,
    "input_text"     : "",     # persists the user's last spec text
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset_run():
    st.session_state.console_lines  = []
    st.session_state.design_result  = None
    st.session_state.design_running = False
    st.session_state.design_done    = False


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

_MAX_LINES = 700

def _append_console(text: str):
    lines = st.session_state.console_lines
    lines.append(text)
    if len(lines) > _MAX_LINES:
        del lines[:len(lines) - _MAX_LINES]

def _render_console(ph):
    content = "".join(st.session_state.console_lines)
    content = content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    ph.markdown(f'<div class="console-wrap">{content}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy cached assistant
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AircraftDesignAssistant…")
def _get_assistant():
    from AircraftDesignAssistant import AircraftDesignAssistant
    return AircraftDesignAssistant()


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(spec_text: str, result_box: list, err_box: list,
            q: "queue.Queue[str | None]"):
    _install_capture(q)
    try:
        result = _get_assistant().start(spec_text)
        result_box.append(result)
    except Exception:
        import traceback
        err_box.append(traceback.format_exc())
    finally:
        q.put(None)
        _restore_stdout()


# ─────────────────────────────────────────────────────────────────────────────
# Result rendering
# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return (text.replace("&","&amp;").replace("<","&lt;")
                .replace(">","&gt;").replace("\n","<br>"))

def _find(text: str, pattern: str, default: str = "—") -> str:
    import re
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else default


def _render_metric_cards(display_text: str):
    st.subheader("📊 Key Design Parameters")

    wing_area = _find(display_text, r"Area:\s*([\d.]+)\s*m²")
    ar        = _find(display_text, r"\bAR[: ]+([0-9.]+)")
    span      = _find(display_text, r"Span:\s*([\d.]+)\s*m")
    mac       = _find(display_text, r"MAC:\s*([\d.]+)\s*m")
    taper     = _find(display_text, r"Taper:\s*([\d.]+)")
    sweep     = _find(display_text, r"Sweep:\s*([\d.]+)")
    ld        = _find(display_text, r"L/D \(3-D\)\s*:\s*([\d.]+)")
    cl        = _find(display_text, r"CL_cruise:\s*([\d.]+)")
    cd        = _find(display_text, r"CD_total\s*:\s*([\d.]+)")
    v_stall   = _find(display_text, r"Stall speed:.*?\(([\d.]+)\s*km/h\)")
    pwr       = _find(display_text, r"Power req\s*:\s*([\d.]+)\s*kW")
    thrust    = _find(display_text, r"Thrust req\s*:\s*([\d.]+)\s*N")
    ht_vol    = _find(display_text, r"Vh:\s*([\d.]+)")
    vt_vol    = _find(display_text, r"Vv:\s*([\d.]+)")
    ht_area   = _find(display_text, r"Horizontal Tail.*?Area:\s*([\d.]+)\s*m²")
    wing_foil = _find(display_text, r"Wing\s*[—–\-]\s*([A-Za-z0-9\-]+)")
    tail_foil = _find(display_text, r"Horizontal Tail\s*[—–\-]\s*([A-Za-z0-9\-]+)")
    prop_note = _find(display_text, r"Propulsion\s*\n\s*(.+?)(?:\n|$)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Wing Geometry**")
        st.metric("Wing Area",    f"{wing_area} m²")
        st.metric("Aspect Ratio", ar)
        st.metric("Span",         f"{span} m")
        st.metric("MAC",          f"{mac} m")
        st.metric("Taper Ratio",  taper)
        st.metric("Sweep",        f"{sweep}°")

    with c2:
        st.markdown("**Aerodynamics**")
        st.metric("Cruise L/D",   ld)
        st.metric("CL cruise",    cl)
        st.metric("CD total",     cd)
        st.metric("Stall Speed",  f"{v_stall} km/h")

    with c3:
        st.markdown("**Propulsion & Tail**")
        st.metric("Power Req.",       f"{pwr} kW")
        st.metric("Thrust Req.",      f"{thrust} N")
        st.metric("HT Area",          f"{ht_area} m²")
        st.metric("HT Vol. Coeff.",   ht_vol)
        st.metric("VT Vol. Coeff.",   vt_vol)

    with c4:
        st.markdown("**Selected Airfoils**")
        st.metric("Wing Airfoil", wing_foil)
        st.metric("Tail Airfoil", tail_foil)
        if prop_note != "—":
            st.markdown(f"**Propulsion note**")
            st.caption(prop_note[:120])


def _render_results(result: Any):
    st.markdown("---")
    st.header("🗂️ Design Results")

    if isinstance(result, str):
        # Could be a validation prompt asking for more info
        st.warning(result)
        return

    if isinstance(result, dict) and "error" in result:
        st.error("Pipeline error — see console above for the full traceback.")
        with st.expander("Traceback"):
            st.code(result["error"])
        return

    display_text = ""
    if isinstance(result, dict):
        display_text = result.get("display", "")
    elif isinstance(result, str):
        display_text = result

    if display_text:
        _render_metric_cards(display_text)
        st.markdown("---")

        col_w, col_t = st.columns(2)
        wing_sel = _find(display_text, r"✅ Selected: ([A-Za-z0-9\-]+)")
        tail_sel = _find(display_text,
                         r"✅ Selected: [A-Za-z0-9\-]+.*?✅ Selected: ([A-Za-z0-9\-]+)", "—")
        with col_w:
            st.subheader("🛩️ Wing Airfoil")
            st.success(f"**{wing_sel}**")
        with col_t:
            st.subheader("🔩 Tail Airfoil")
            st.success(f"**{tail_sel}**")

        st.markdown("---")
        st.subheader("📄 Full Structured Report")
        st.markdown(f'<div class="result-block">{_esc(display_text)}</div>',
                    unsafe_allow_html=True)

    if isinstance(result, dict) and result.get("streaming"):
        st.markdown("---")
        st.info(
            "📡 **AI Engineering Assessment (§8)** was streamed by the LLM "
            "during pipeline execution. The full narrative is in the console log above."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✈️ Aircraft Design")
    st.markdown("""
**Pipeline stages**

`0` Parse & validate requirements (LLM)
`1` Literature + reference search
`1b` Airfoil candidate selection (LLM)
`2a` Wing — XFOIL / Hess-Smith panel
`2b` Tail — XFOIL / Hess-Smith panel
`3` Preliminary sizing
`3b` Engineering constraint search
`4` AI design assessment (LLM)
""")
    st.markdown("---")
    st.markdown("""
**XFOIL**

| Environment | Analysis |
|---|---|
| Windows + `xfoil.exe` | ✅ Viscous |
| Linux / cloud | ⚠️ Panel fallback |
| Docker + gfortran | ✅ Viscous |
| `pip install xfoil` | ✅ Viscous |

Binary auto-detected; falls back gracefully.
""")
    st.markdown("---")
    st.markdown("""
**LLM backend**
Local Ollama (`qwen3` / `phi3`).  
Run `ollama serve` before launching.
""")
    if st.session_state.design_done:
        st.markdown("---")
        if st.button("🔄 New Design"):
            _reset_run()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("✈️ Aircraft Design Assistant")
st.markdown(
    "Describe your aircraft in plain English — the AI extracts every parameter "
    "automatically and runs the full design pipeline."
)

# ── Example prompts ───────────────────────────────────────────────────────────
EXAMPLES = [
    "Design a 25 kg fixed-wing surveillance UAV for 6-hour endurance at 120 km/h, "
    "500 m altitude, electric propulsion, 5 kg payload, 4 m span limit, stall below 60 km/h.",

    "Design a 1200 kg two-seat piston trainer for 450 km range at 220 km/h cruise, "
    "1500 m altitude, stall below 90 km/h, aerobatic capability.",

    "Design a 350 kg cargo UAV with 80 kg payload, 600 km range at 160 km/h, "
    "piston engine, 3000 m cruise altitude, 6 m span constraint.",

    "Design a 480 kg electric sailplane for 8-hour thermalling at 100 km/h cruise, "
    "1000 m altitude, 18 m span, stall below 70 km/h.",
]

st.markdown('<p class="section-label">Quick-start examples — click to load</p>',
            unsafe_allow_html=True)

ex_cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(ex_cols, EXAMPLES)):
    with col:
        label = ex.split(",")[0][:42] + "…"
        if st.button(label, key=f"ex_{i}"):
            st.session_state.spec_input = ex   # ← THIS updates the text_area
            st.session_state.input_text = ex   # (optional sync for persistence)
            st.rerun()

st.markdown("---")

# ── Single input box ──────────────────────────────────────────────────────────
spec_text = st.text_area(
    "✏️  Describe your aircraft requirements",
    value=st.session_state.input_text,
    height=160,
    placeholder=(
        "Example:  Design a 25 kg fixed-wing UAV for 6-hour endurance at 120 km/h, "
        "500 m altitude, electric propulsion, 5 kg payload, 4 m span, stall below 60 km/h.\n\n"
        "Include as many of these as you can:\n"
        "  aircraft type · take-off mass · payload · cruise speed · stall limit\n"
        "  range or endurance · cruise altitude · span constraint · propulsion type"
    ),
    key="spec_input",
)

# Keep session copy in sync
st.session_state.input_text = spec_text

run_clicked = st.button("🚀  Run Full Design Pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if run_clicked and not st.session_state.design_running:
    if not spec_text.strip():
        st.warning("Please describe your aircraft requirements before running.")
        st.stop()

    _reset_run()
    st.session_state.design_running = True

    _q: queue.Queue   = queue.Queue()
    _result_box: list = []
    _err_box:    list = []

    t = threading.Thread(
        target=_worker,
        args=(spec_text.strip(), _result_box, _err_box, _q),
        daemon=True,
    )
    t.start()

    # ── Live progress ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("⚙️  Pipeline Progress")

    _STAGES = [
        ("stage 0",   "Parsing requirements",          0.07),
        ("stage 1 —", "Literature search",             0.18),
        ("stage 1b",  "Airfoil candidate selection",   0.27),
        ("stage 2a",  "Wing airfoil analysis",         0.43),
        ("stage 2b",  "Tail airfoil analysis",         0.58),
        ("stage 3 —", "Preliminary sizing",            0.72),
        ("stage 3b",  "Engineering constraints",       0.84),
        ("stage 4",   "AI design assessment",          0.94),
        ("complete",  "Complete",                      1.00),
    ]

    progress_bar   = st.progress(0.0)
    status_empty   = st.empty()
    st.markdown('<p class="section-label">Backend Console</p>',
                unsafe_allow_html=True)
    console_ph     = st.empty()

    cur_frac  = 0.0
    cur_label = _STAGES[0][1]
    status_empty.info(f"⚙️  {cur_label} …")

    # Poll loop
    while t.is_alive() or not _q.empty():
        changed = False
        while True:
            try:
                chunk = _q.get_nowait()
            except queue.Empty:
                break
            if chunk is None:
                break
            _append_console(chunk)
            changed = True
            lower = chunk.lower()
            for kw, label, frac in _STAGES:
                if kw in lower and frac > cur_frac:
                    cur_frac  = frac
                    cur_label = label
                    break

        if changed:
            progress_bar.progress(min(cur_frac, 0.97))
            status_empty.info(f"⚙️  {cur_label} …")
            _render_console(console_ph)

        time.sleep(0.12)

    # Final drain
    while True:
        try:
            chunk = _q.get_nowait()
        except queue.Empty:
            break
        if chunk is not None:
            _append_console(chunk)

    _render_console(console_ph)
    progress_bar.progress(1.0)

    if _err_box:
        status_empty.error("❌  Pipeline error — see console for traceback.")
        st.session_state.design_result = {"error": _err_box[0]}
    elif _result_box:
        status_empty.success("✅  Design pipeline complete!")
        st.session_state.design_result = _result_box[0]
    else:
        status_empty.warning("⚠️  Pipeline finished with no result returned.")

    st.session_state.design_done    = True
    st.session_state.design_running = False

    _render_results(st.session_state.design_result)

# ─────────────────────────────────────────────────────────────────────────────
# Persisted view (subsequent reruns / sidebar interactions)
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.design_done and not run_clicked:
    st.markdown("---")
    st.header("⚙️  Last Run — Console Log")
    _render_console(st.empty())
    if st.session_state.design_result is not None:
        _render_results(st.session_state.design_result)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="color:#1f2937;font-size:.72rem;text-align:center;letter-spacing:.08em;">'
    'AIRCRAFT DESIGN ASSISTANT &nbsp;·&nbsp; '
    'XFOIL 6.99 / Hess-Smith Panel &nbsp;·&nbsp; '
    'Ollama LLM'
    '</p>',
    unsafe_allow_html=True,
)
