import streamlit as st
import requests
import base64
import time

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "https://datamindbackend-production.up.railway.app/agent/invoke"
HEALTH_URL  = "https://datamindbackend-production.up.railway.app/health"
MAX_FILE_MB     = 10
REQUEST_TIMEOUT = 400

st.set_page_config(page_title="DataMind AI", page_icon="◈", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Neon Terminal × Brutalist Data Lab aesthetic
# Dark background, electric accent, mono typography, sharp edges
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg:       #0a0a0f;
  --surface:  #111118;
  --panel:    #16161f;
  --border:   #242430;
  --border2:  #2e2e3e;
  --text:     #e2e2f0;
  --muted:    #5a5a78;
  --dim:      #30303f;
  --cyan:     #00e5ff;
  --cyan-dim: #0097aa;
  --green:    #00ff9d;
  --red:      #ff4060;
  --amber:    #ffb020;
  --magenta:  #e040fb;
  --glow-c:   rgba(0,229,255,0.12);
  --glow-g:   rgba(0,255,157,0.10);
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]  { display: none; }
.main .block-container     { max-width: 1400px; padding: 0 2rem 5rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar         { width: 4px; height: 4px; }
::-webkit-scrollbar-track   { background: var(--bg); }
::-webkit-scrollbar-thumb   { background: var(--border2); border-radius: 4px; }

/* ══════════════════════════════════════════════════════════════
   HEADER
══════════════════════════════════════════════════════════════ */
.dm-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  padding: 2.2rem 0 1.4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
  position: relative;
}

.dm-header::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0;
  width: 120px; height: 2px;
  background: var(--cyan);
  box-shadow: 0 0 12px var(--cyan);
}

.dm-logo {
  font-family: 'Syne', sans-serif;
  font-size: 2.8rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.04em;
  color: var(--text);
}
.dm-logo span { color: var(--cyan); text-shadow: 0 0 20px var(--cyan); }

.dm-tagline {
  font-size: 0.6rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.35rem;
  font-family: 'Space Mono', monospace;
}

.dm-status {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 0.35rem 0.9rem;
  border-radius: 2px;
  border: 1px solid;
}
.dm-status.online  { color: var(--green); border-color: var(--green); background: rgba(0,255,157,0.06); text-shadow: 0 0 8px var(--green); }
.dm-status.offline { color: var(--red);   border-color: var(--red);   background: rgba(255,64,96,0.06); }

/* ══════════════════════════════════════════════════════════════
   SECTION LABELS
══════════════════════════════════════════════════════════════ */
.sec-label {
  font-size: 0.6rem;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--cyan-dim);
  font-family: 'Space Mono', monospace;
  margin-bottom: 0.6rem;
  display: flex;
  align-items: center;
  gap: 0.8rem;
}
.sec-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ══════════════════════════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: 4px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--cyan-dim) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzone"] span {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.72rem !important;
  color: var(--muted) !important;
}

.file-badge {
  font-family: 'Space Mono', monospace;
  font-size: 0.68rem;
  padding: 0.4rem 0.9rem;
  border-radius: 3px;
  margin-top: 0.5rem;
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  border: 1px solid;
}
.file-badge.ok  { color: var(--green); border-color: var(--green); background: rgba(0,255,157,0.05); }
.file-badge.err { color: var(--red);   border-color: var(--red);   background: rgba(255,64,96,0.05); }

/* ══════════════════════════════════════════════════════════════
   TEXT AREA
══════════════════════════════════════════════════════════════ */
.stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 4px !important;
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.82rem !important;
  padding: 0.9rem 1rem !important;
  line-height: 1.6 !important;
  caret-color: var(--cyan) !important;
}
.stTextArea textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px var(--glow-c) !important;
  outline: none !important;
}
.stTextArea label {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.6rem !important;
  letter-spacing: 0.3em !important;
  text-transform: uppercase !important;
  color: var(--cyan-dim) !important;
}

/* ══════════════════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════════════════ */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
  padding: 0.75rem 1.8rem !important;
  border-radius: 3px !important;
  transition: all .18s ease !important;
  width: 100% !important;
}
.stButton > button:hover {
  background: var(--glow-c) !important;
  box-shadow: 0 0 16px var(--glow-c) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
  border-color: var(--dim) !important;
  color: var(--muted) !important;
  box-shadow: none !important;
}

/* Secondary / danger button variant */
.stButton.danger > button {
  border-color: var(--red) !important;
  color: var(--red) !important;
}
.stButton.danger > button:hover {
  background: rgba(255,64,96,0.08) !important;
}

/* ══════════════════════════════════════════════════════════════
   TURN CARDS
══════════════════════════════════════════════════════════════ */
.turn-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--cyan);
  border-radius: 4px;
  margin-bottom: 1.2rem;
  overflow: hidden;
}
.turn-card.latest { border-left-color: var(--green); box-shadow: 0 0 20px var(--glow-g); }
.turn-card.errored { border-left-color: var(--red); }

.turn-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.7rem 1.1rem;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  cursor: pointer;
}
.turn-header:hover { background: #1c1c28; }

.turn-num {
  font-size: 0.58rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--cyan);
  font-family: 'Space Mono', monospace;
}
.turn-num.errored { color: var(--red); }

.turn-query {
  font-family: 'Syne', sans-serif;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text);
  flex: 1;
  padding: 0 1rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.turn-chevron { font-size: 0.6rem; color: var(--muted); }

.turn-body { padding: 1.2rem 1.1rem; }

/* ══════════════════════════════════════════════════════════════
   CODE BLOCK
══════════════════════════════════════════════════════════════ */
.code-chrome {
  background: #0d1117;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 1rem;
}
.code-chrome-bar {
  background: #161b22;
  padding: 0.5rem 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.mac-dot { width: 10px; height: 10px; border-radius: 50%; }
.mac-r { background: #ff5f57; }
.mac-y { background: #febc2e; }
.mac-g { background: #28c840; }
.code-chrome-label {
  margin-left: auto;
  font-family: 'Space Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
}

/* ══════════════════════════════════════════════════════════════
   CONSOLE OUTPUT
══════════════════════════════════════════════════════════════ */
.console {
  background: #050508;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem 1.1rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.73rem;
  color: var(--green);
  line-height: 1.8;
  min-height: 60px;
  white-space: pre-wrap;
  word-break: break-word;
  text-shadow: 0 0 6px rgba(0,255,157,0.3);
}
.console.empty { color: var(--muted); font-style: italic; text-shadow: none; }

/* ══════════════════════════════════════════════════════════════
   IMAGE CARD
══════════════════════════════════════════════════════════════ */
.img-frame {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem;
}

/* ══════════════════════════════════════════════════════════════
   ERROR / WARN BANNERS
══════════════════════════════════════════════════════════════ */
.err-banner {
  background: rgba(255,64,96,0.07);
  border: 1px solid rgba(255,64,96,0.35);
  border-left: 3px solid var(--red);
  border-radius: 4px;
  padding: 0.8rem 1rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.73rem;
  color: var(--red);
  line-height: 1.6;
  margin-bottom: 0.8rem;
}
.err-label {
  font-size: 0.58rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.25rem;
  opacity: 0.7;
}

.warn-banner {
  background: rgba(255,176,32,0.07);
  border: 1px solid rgba(255,176,32,0.3);
  border-left: 3px solid var(--amber);
  border-radius: 4px;
  padding: 0.7rem 1rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  color: var(--amber);
  margin-bottom: 1rem;
}

/* ══════════════════════════════════════════════════════════════
   MEMORY BADGE  (session turns counter)
══════════════════════════════════════════════════════════════ */
.mem-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  color: var(--magenta);
  border: 1px solid rgba(224,64,251,0.3);
  background: rgba(224,64,251,0.06);
  padding: 0.3rem 0.8rem;
  border-radius: 2px;
  margin-top: 0.4rem;
}

/* ══════════════════════════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════════════════════════ */
.empty-state {
  background: var(--surface);
  border: 1px dashed var(--border2);
  border-radius: 4px;
  padding: 4rem 2rem;
  text-align: center;
  font-family: 'Space Mono', monospace;
}
.empty-state .glyph {
  font-size: 2.8rem;
  opacity: 0.08;
  margin-bottom: 1rem;
  display: block;
  font-family: 'Syne', sans-serif;
}
.empty-state .hint {
  font-size: 0.7rem;
  color: var(--muted);
  line-height: 2;
}
.empty-state .hint b { color: var(--cyan); }

/* ══════════════════════════════════════════════════════════════
   STEP TRACKER
══════════════════════════════════════════════════════════════ */
.step-track { display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 1.2rem; }
.step-item {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  padding: 0.65rem 1rem;
  border-radius: 3px;
  border: 1px solid var(--border);
  background: var(--surface);
  font-family: 'Space Mono', monospace;
  font-size: 0.73rem;
  transition: all .2s ease;
}
.step-item.active  { border-color: var(--cyan);  background: var(--glow-c); color: var(--cyan); }
.step-item.done    { border-color: var(--border); color: var(--muted); }
.step-item.error   { border-color: var(--red);    background: rgba(255,64,96,0.07); color: var(--red); }
.step-item.waiting { opacity: 0.3; }
.step-icon { flex-shrink: 0; width: 1.1rem; text-align: center; }
.step-detail { font-size: 0.65rem; margin-top: 0.15rem; opacity: 0.7; }

/* Download button */
[data-testid="stDownloadButton"] button {
  background: transparent !important;
  border: 1px solid var(--border2) !important;
  color: var(--muted) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.12em !important;
  border-radius: 3px !important;
  padding: 0.45rem 1rem !important;
  margin-top: 0.6rem !important;
}
[data-testid="stDownloadButton"] button:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
}

[data-testid="stHorizontalBlock"] { gap: 2rem !important; }

/* Expander */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),   # sent to backend each request
    ("turns",        []),   # UI display history
    ("last_file",    None), # detect file change → reset
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_health() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def validate_file(f) -> str | None:
    if f is None:                          return "No file selected."
    if f.size / 1024 / 1024 > MAX_FILE_MB: return f"File exceeds {MAX_FILE_MB} MB limit."
    if not f.name.lower().endswith(".csv"):return "Only CSV files are supported."
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
online = check_health()
sc     = "online"  if online else "offline"
sl     = "● LIVE"  if online else "● OFFLINE"
turns  = len(st.session_state.turns)

st.markdown(f"""
<div class="dm-header">
  <div>
    <div class="dm-logo">Data<span>Mind</span></div>
    <div class="dm-tagline">AI · Data Intelligence · Conversational Analysis</div>
  </div>
  <div style="text-align:right">
    <div class="dm-status {sc}">{sl}</div>
    {"" if online else '<div style="font-size:.62rem;color:var(--red);margin-top:.4rem;font-family:Space Mono,monospace">run: python backend.py</div>'}
  </div>
</div>
""", unsafe_allow_html=True)

if not online:
    st.markdown('<div class="warn-banner">⚠ Backend is offline. Start it with <b>python backend.py</b> then refresh.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT PANEL
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.7], gap="large")

with col_left:
    st.markdown('<div class="sec-label">01 · Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded:
        mb  = uploaded.size / 1024 / 1024
        ok  = mb <= MAX_FILE_MB
        cls = "ok" if ok else "err"
        ico = "✓" if ok else "✕"
        st.markdown(
            f'<div class="file-badge {cls}">'
            f'<span>{ico}</span><span>{uploaded.name}</span>'
            f'<span style="opacity:.5">·</span><span>{mb:.2f} MB</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Auto-reset history on new file
        if st.session_state.last_file != uploaded.name:
            st.session_state.chat_history = []
            st.session_state.turns        = []
            st.session_state.last_file    = uploaded.name

with col_right:
    st.markdown('<div class="sec-label">02 · Query</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Query",
        placeholder=(
            "e.g.  Which country has the highest sales?\n"
            "Then: Now show me a bar chart of the top 5.\n"
            "Then: Filter to Q3 only and compare by region."
        ),
        height=128,
        label_visibility="collapsed",
    )
    if turns > 0:
        st.markdown(
            f'<div class="mem-badge">◈ {turns} turn{"s" if turns!=1 else ""} in memory · context active</div>',
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

# Buttons row
b1, b2, b3 = st.columns([5, 1.5, 1.5], gap="small")
with b1:
    st.markdown('<div class="sec-label">03 · Execute</div>', unsafe_allow_html=True)
    run = st.button("◈  Run Analysis", use_container_width=True, disabled=not online)
with b2:
    st.markdown('<div class="sec-label">Session</div>', unsafe_allow_html=True)
    clear = st.button("↺  New Session", use_container_width=True)
with b3:
    st.markdown("<div style='height:1.45rem'></div>", unsafe_allow_html=True)
    # spacer to align with button
    st.markdown("")

if clear:
    st.session_state.chat_history = []
    st.session_state.turns        = []
    st.session_state.last_file    = None
    st.rerun()

st.markdown("<hr style='border-color:var(--border);margin:1.5rem 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AGENT RUN
# ══════════════════════════════════════════════════════════════════════════════
if run:
    # ── Validate ──────────────────────────────────────────────────────────────
    ferr = validate_file(uploaded)
    if ferr:
        st.markdown(f'<div class="err-banner"><div class="err-label">Input Error</div>{ferr}</div>', unsafe_allow_html=True)
        st.stop()
    if not query or not query.strip():
        st.markdown('<div class="err-banner"><div class="err-label">Input Error</div>Please enter a query.</div>', unsafe_allow_html=True)
        st.stop()

    # ── Step tracker UI ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Processing Log</div>', unsafe_allow_html=True)
    s1 = st.empty(); s2 = st.empty(); s3 = st.empty()
    s4 = st.empty(); s5 = st.empty()

    def step(icon, num, name, detail, state="waiting"):
        cls = f"step-item {state}"
        return (
            f'<div class="{cls}">'
            f'<span class="step-icon">{icon}</span>'
            f'<div><div>{num} &nbsp; <b>{name}</b></div>'
            f'<div class="step-detail">{detail}</div></div></div>'
        )

    s1.markdown(step("⚙", "01", "Encoding File",      "Base64 encoding CSV…",        "active"),  unsafe_allow_html=True)
    s2.markdown(step("◌", "02", "Backend Request",    "Waiting…",                    "waiting"), unsafe_allow_html=True)
    s3.markdown(step("◌", "03", "Data Extractor",     "Waiting…",                    "waiting"), unsafe_allow_html=True)
    s4.markdown(step("◌", "04", "Code Generator",     f"LLM · {len(st.session_state.chat_history)//2} turns of context", "waiting"), unsafe_allow_html=True)
    s5.markdown(step("◌", "05", "E2B Sandbox",        "Waiting…",                    "waiting"), unsafe_allow_html=True)

    time.sleep(0.2)

    # Encode file
    raw    = uploaded.getvalue()
    b64    = base64.b64encode(raw).decode("utf-8")
    kb     = len(raw) / 1024

    s1.markdown(step("✓", "01", "File Encoded",       f"{kb:.1f} KB ready",          "done"),    unsafe_allow_html=True)
    s2.markdown(step("⚙", "02", "Backend Request",    "Sending to FastAPI…",         "active"),  unsafe_allow_html=True)
    s3.markdown(step("⚙", "03", "Data Extractor",     "Parsing CSV…",                "active"),  unsafe_allow_html=True)
    s4.markdown(step("⚙", "04", "Code Generator",     "LLM generating code…",        "active"),  unsafe_allow_html=True)
    s5.markdown(step("⚙", "05", "E2B Sandbox",        "Spinning up sandbox…",        "active"),  unsafe_allow_html=True)

    # Payload — same contract as original, now includes chat_history
    payload = {
        "input": {
            "user_query":         query.strip(),
            "file_data":          b64,
            "extracted_data":     "",
            "code_generated_llm": "",
            "code_output":        "",
            "image_data":         "",
            "code_error":         "",
            "chat_history":       st.session_state.chat_history,   # ← memory
        }
    }

    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=REQUEST_TIMEOUT)
        s2.markdown(step("✓", "02", "Backend Connected",  f"HTTP {resp.status_code}", "done"), unsafe_allow_html=True)

        if resp.status_code == 200:
            out       = resp.json().get("output", {})
            code_v    = out.get("code_generated_llm", "")
            console_v = out.get("code_output",        "")
            img_v     = out.get("image_data",         "")
            err_v     = out.get("code_error",         "")

            # Update local history from backend response
            st.session_state.chat_history = out.get("chat_history", st.session_state.chat_history)

            s3.markdown(step("✓", "03", "Data Extractor",  "CSV parsed successfully",                    "done"),  unsafe_allow_html=True)
            s4.markdown(step("✓" if code_v else "✕", "04", "Code Generator", f"{len(code_v)} chars generated" if code_v else "No code", "done" if code_v else "error"), unsafe_allow_html=True)
            if err_v:
                s5.markdown(step("✕", "05", "E2B Sandbox", f"Error: {err_v[:80]}", "error"), unsafe_allow_html=True)
            else:
                s5.markdown(step("✓", "05", "E2B Sandbox", f"Done · {'chart ✓' if img_v else 'no chart'} · {len(console_v)} chars stdout", "done"), unsafe_allow_html=True)

            # Store turn for display
            st.session_state.turns.append({
                "n":       len(st.session_state.turns) + 1,
                "query":   query.strip(),
                "code":    code_v,
                "console": console_v,
                "image":   img_v,
                "error":   err_v,
            })
            st.rerun()

        else:
            for sx in [s3, s4, s5]:
                sx.markdown(step("✕", "—", "Skipped", "Backend error", "error"), unsafe_allow_html=True)
            try:    err = resp.json().get("error", resp.text)
            except: err = resp.text
            st.markdown(f'<div class="err-banner"><div class="err-label">Backend Error {resp.status_code}</div>{err}</div>', unsafe_allow_html=True)

    except requests.exceptions.ConnectionError:
        s2.markdown(step("✕", "02", "Connection Failed", "Cannot reach backend",   "error"), unsafe_allow_html=True)
        for sx in [s3, s4, s5]:
            sx.markdown(step("✕", "—", "Skipped", "No connection", "error"), unsafe_allow_html=True)
    except requests.exceptions.Timeout:
        s5.markdown(step("✕", "05", "E2B Sandbox", "Request timed out — try simpler query", "error"), unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="err-banner"><div class="err-label">Unexpected Error</div>{e}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.turns:
    st.markdown('<div class="sec-label">Conversation History</div>', unsafe_allow_html=True)

    for turn in reversed(st.session_state.turns):
        n  = turn["n"]
        q  = turn["query"]
        is_latest = (n == len(st.session_state.turns))
        has_error = bool(turn["error"])

        card_cls = "turn-card latest" if is_latest else ("turn-card errored" if has_error else "turn-card")
        num_cls  = "turn-num errored" if has_error else "turn-num"
        label    = "TURN" if not has_error else "ERROR"

        with st.expander(
            f"Turn {n}  ·  {q[:70]}{'…' if len(q)>70 else ''}",
            expanded=is_latest,
        ):
            if turn["error"]:
                st.markdown(
                    f'<div class="err-banner"><div class="err-label">Execution Error</div>{turn["error"]}</div>',
                    unsafe_allow_html=True,
                )

            col_code, col_vis = st.columns([1, 1], gap="large")

            # ── LEFT: Code + Console ──────────────────────────────────────────
            with col_code:
                st.markdown(
                    '<div style="font-family:Space Mono,monospace;font-size:.58rem;'
                    'letter-spacing:.25em;text-transform:uppercase;color:var(--cyan-dim);'
                    'margin-bottom:.5rem">Generated Code</div>',
                    unsafe_allow_html=True,
                )
                if turn["code"]:
                    st.markdown(
                        '<div class="code-chrome-bar" style="background:#161b22;padding:.45rem .9rem;'
                        'display:flex;align-items:center;gap:.45rem;border-radius:4px 4px 0 0;'
                        'border:1px solid var(--border);border-bottom:none">'
                        '<span class="mac-dot mac-r" style="width:9px;height:9px;border-radius:50%;background:#ff5f57;display:inline-block"></span>'
                        '<span class="mac-dot mac-y" style="width:9px;height:9px;border-radius:50%;background:#febc2e;display:inline-block"></span>'
                        '<span class="mac-dot mac-g" style="width:9px;height:9px;border-radius:50%;background:#28c840;display:inline-block"></span>'
                        '<span style="margin-left:auto;font-family:Space Mono,monospace;font-size:.55rem;'
                        'letter-spacing:.2em;color:var(--muted)">ANALYSIS.PY</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.code(turn["code"], language="python")
                else:
                    st.markdown(
                        '<div class="console empty" style="font-family:Space Mono,monospace;'
                        'font-size:.72rem;color:var(--muted);font-style:italic;background:#050508;'
                        'border:1px solid var(--border);border-radius:4px;padding:.9rem">No code generated.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div style="font-family:Space Mono,monospace;font-size:.58rem;'
                    'letter-spacing:.25em;text-transform:uppercase;color:var(--cyan-dim);'
                    'margin:.9rem 0 .5rem">Console Output</div>',
                    unsafe_allow_html=True,
                )
                if turn["console"]:
                    st.markdown(
                        f'<div class="console">{turn["console"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="console" style="color:var(--muted);font-style:italic;'
                        'font-family:Space Mono,monospace;font-size:.72rem;background:#050508;'
                        'border:1px solid var(--border);border-radius:4px;padding:.9rem">No output printed.</div>',
                        unsafe_allow_html=True,
                    )

            # ── RIGHT: Visualization ──────────────────────────────────────────
            with col_vis:
                st.markdown(
                    '<div style="font-family:Space Mono,monospace;font-size:.58rem;'
                    'letter-spacing:.25em;text-transform:uppercase;color:var(--cyan-dim);'
                    'margin-bottom:.5rem">Visualization</div>',
                    unsafe_allow_html=True,
                )
                if turn["image"]:
                    img_bytes = base64.b64decode(turn["image"])
                    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                    st.image(img_bytes, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.download_button(
                        f"↓  Download Chart — Turn {n}",
                        data=img_bytes,
                        file_name=f"datamind_chart_turn{n}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{n}",
                    )
                else:
                    st.markdown(
                        '<div class="empty-state" style="padding:3rem 1.5rem">'
                        '<span class="glyph">◈</span>'
                        '<div class="hint">No chart generated.<br>'
                        '<b>Ask for a bar chart, line graph, scatter plot…</b></div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

else:
    st.markdown(
        '<div class="empty-state">'
        '<span class="glyph">◈</span>'
        '<div class="hint">'
        'Upload a CSV · Enter a query · Press <b>Run Analysis</b><br><br>'
        'Ask follow-up questions — the agent remembers context.<br>'
        '<span style="color:var(--muted)">e.g. "Now show the top 10" · "Filter by region" · "Add a trend line"</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
