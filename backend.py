import os
import base64
import io
import time
import logging
from pathlib import Path
from typing import Optional, TypedDict, List
from functools import wraps

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
E2B_API_KEY  = os.getenv("E2B_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY missing from .env")
if not E2B_API_KEY:
    raise EnvironmentError("E2B_API_KEY missing from .env")

log.info("✅ API keys loaded")

# ── Library imports ───────────────────────────────────────────────────────────
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from e2b_code_interpreter import Sandbox   # pip install e2b-code-interpreter
import uvicorn

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_MB           = 10
MAX_FILE_BYTES        = MAX_FILE_MB * 1024 * 1024
FILE_PATH_IN_SANDBOX  = "/app/data.csv"
PREVIEW_MAX_CHARS     = 1500
PREVIEW_MAX_ROWS      = 3
PREVIEW_MAX_COLS      = 15
MAX_HISTORY_TURNS     = 6   # keep last 6 exchanges (12 messages) for token budget

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    user_query:         str
    file_data:          str
    extracted_data:     str
    code_generated_llm: str
    code_output:        Optional[str]
    image_data:         Optional[str]
    code_error:         Optional[str]
    chat_history:       List[dict]   # {"role": "human"|"ai", "content": "..."}

# ── Helpers ───────────────────────────────────────────────────────────────────
def retry(max_attempts=3, delay=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    log.warning(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_err
        return wrapper
    return decorator

def safe_str(value) -> str:
    if value is None:           return ""
    if isinstance(value, list): return "\n".join(str(i) for i in value)
    if isinstance(value, bytes):return value.decode("utf-8", errors="replace")
    return str(value)

def history_to_messages(history: list) -> list:
    """Convert stored history dicts → LangChain message objects (trimmed)."""
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    msgs = []
    for m in trimmed:
        if m.get("role") == "human":
            msgs.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "ai":
            msgs.append(AIMessage(content=m["content"]))
    return msgs

def update_history(history: list, user_msg: str, ai_msg: str) -> list:
    updated = list(history)
    updated.append({"role": "human", "content": user_msg})
    updated.append({"role": "ai",    "content": ai_msg})
    if len(updated) > MAX_HISTORY_TURNS * 2:
        updated = updated[-(MAX_HISTORY_TURNS * 2):]
    return updated


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Data Extractor
# ══════════════════════════════════════════════════════════════════════════════
def DataExtractor(state: State):
    log.info("NODE 1 | DataExtractor")
    file_b64 = state["file_data"]
    history  = state.get("chat_history", [])

    if not file_b64:
        return {"code_error": "No file uploaded.", "chat_history": history}

    try:
        file_bytes = base64.b64decode(file_b64)

        if len(file_bytes) > MAX_FILE_BYTES:
            mb = len(file_bytes) / (1024 * 1024)
            return {"code_error": f"File too large ({mb:.1f} MB). Max is {MAX_FILE_MB} MB.", "chat_history": history}

        df = None
        for enc in ["utf-8", "latin-1", "windows-1252", "iso-8859-1"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                break
            except Exception:
                continue

        if df is None:
            return {"code_error": "Cannot parse CSV file.", "chat_history": history}

        cols       = df.columns.tolist()
        preview_df = df.head(PREVIEW_MAX_ROWS)
        col_note   = ""

        if len(cols) > PREVIEW_MAX_COLS:
            preview_df = preview_df.iloc[:, :PREVIEW_MAX_COLS]
            col_note   = f"\n(showing {PREVIEW_MAX_COLS} of {len(cols)} columns)"

        preview = preview_df.to_string()
        if len(preview) > PREVIEW_MAX_CHARS:
            preview = preview[:PREVIEW_MAX_CHARS] + "\n...(truncated)"

        extracted = (
            f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n"
            f"Columns: {cols}{col_note}\n\nSample:\n{preview}"
        )
        log.info("NODE 1 | Done")
        return {"extracted_data": extracted, "chat_history": history}

    except Exception as e:
        log.error(f"NODE 1 | Error: {e}")
        return {"code_error": f"Extraction error: {e}", "chat_history": history}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Code Generator  (with conversational memory)
# ══════════════════════════════════════════════════════════════════════════════
@retry(max_attempts=3, delay=3.0)
def _call_llm(llm, system_msg: str, history_msgs: list, human_msg: str) -> str:
    messages = [SystemMessage(content=system_msg)] + history_msgs + [HumanMessage(content=human_msg)]
    return llm.invoke(messages).content


def Code_generator(state: State):
    log.info("NODE 2 | CodeGenerator")
    query   = state["user_query"]
    preview = state["extracted_data"]
    history = state.get("chat_history", [])

    if not query or not query.strip():
        return {"code_error": "No query provided.", "chat_history": history}
    if not preview:
        return {"code_error": "No data preview available.", "chat_history": history}

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=GROQ_API_KEY)

        system_msg = """You are an expert Python Data Analyst with full memory of this conversation.

RULES:
- Dataset is ALWAYS at: /app/data.csv
- ALWAYS load: df = pd.read_csv('/app/data.csv', encoding='latin-1')
- Always print() your final answer so results appear in stdout.
- For plots: use plt.style.use('dark_background'), vibrant colors, tight_layout(), plt.show().
- Import all libraries at the top.
- Use conversation history to answer follow-up questions (e.g. "now show a chart of that").
- Return ONLY raw Python code. Zero markdown. Zero backticks. Zero explanations."""

        history_msgs = history_to_messages(history)
        human_msg    = f"Data:\n{preview}\n\nQuery: {query}"
        log.info(f"NODE 2 | History: {len(history_msgs)} messages")

        raw   = _call_llm(llm, system_msg, history_msgs, human_msg)
        code  = raw.replace("```python", "").replace("```", "").strip()

        log.info(f"NODE 2 | Code: {len(code)} chars")
        return {"code_generated_llm": code, "chat_history": history}

    except Exception as e:
        log.error(f"NODE 2 | Error: {e}")
        err = str(e)
        if "rate_limit" in err.lower():
            return {"code_error": "Rate limit hit. Wait 30s and retry.", "chat_history": history}
        return {"code_error": f"Code generation error: {err}", "chat_history": history}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Code Executor  (E2B Sandbox)
# ══════════════════════════════════════════════════════════════════════════════
def CodeExecutorNode(state: State):
    log.info("NODE 3 | CodeExecutor")
    code     = state["code_generated_llm"]
    file_b64 = state["file_data"]
    query    = state["user_query"]
    history  = state.get("chat_history", [])

    if not code or not code.strip():
        return {"code_output": "", "image_data": "", "code_error": "No code to execute.", "chat_history": history}

    # Normalise file paths the LLM might have used
    for wrong in ["'data.csv'", "/home/data.csv", "/tmp/data.csv"]:
        code = code.replace(wrong, f"'{FILE_PATH_IN_SANDBOX}'")
    code = code.replace('"data.csv"', f'"{FILE_PATH_IN_SANDBOX}"')

    try:
        log.info("NODE 3 | Creating E2B sandbox")
        with Sandbox.create() as sandbox:
            # Upload CSV
            if file_b64:
                sandbox.files.write(FILE_PATH_IN_SANDBOX, base64.b64decode(file_b64))
                log.info("NODE 3 | CSV uploaded to sandbox")

            # Execute
            execution = sandbox.run_code(code, timeout=500)

            if execution.error:
                name  = getattr(execution.error, "name",  "Error")
                value = getattr(execution.error, "value", str(execution.error))
                human = f"Runtime Error — {name}: {value}"
                if "ModuleNotFoundError" in name:
                    human = f"Missing library: {value}"
                elif "FileNotFoundError" in name:
                    human = "Data file not found in sandbox. Try again."
                log.error(f"NODE 3 | {human}")
                new_history = update_history(history, query, f"[Error] {human}")
                return {"code_output": "", "image_data": "", "code_error": human, "chat_history": new_history}

            # Capture chart
            img = ""
            for res in (execution.results or []):
                if hasattr(res, "png") and res.png:
                    img = res.png
                    log.info("NODE 3 | Chart captured")
                    break

            # Capture stdout
            output = safe_str(execution.logs.stdout) if execution.logs and execution.logs.stdout else ""
            log.info(f"NODE 3 | stdout: {len(output)} chars | chart: {bool(img)}")

            ai_summary = (output or "[Executed successfully]") + (" — chart generated" if img else "")
            new_history = update_history(history, query, ai_summary)

            return {"code_output": output, "image_data": img, "code_error": "", "chat_history": new_history}

    except Exception as e:
        err = str(e)
        log.error(f"NODE 3 | Sandbox exception: {err}")
        human = "Timed out." if "timed out" in err.lower() else f"Sandbox error: {err}"
        new_history = update_history(history, query, f"[Error] {human}")
        return {"code_output": "", "image_data": "", "code_error": human, "chat_history": new_history}


# ── Build Graph ───────────────────────────────────────────────────────────────
workflow = StateGraph(State)
workflow.add_node("DataExtractor", DataExtractor)
workflow.add_node("CodeGenerator", Code_generator)
workflow.add_node("CodeExecutor",  CodeExecutorNode)
workflow.add_edge(START,           "DataExtractor")
workflow.add_edge("DataExtractor", "CodeGenerator")
workflow.add_edge("CodeGenerator", "CodeExecutor")
workflow.add_edge("CodeExecutor",  END)
graph = workflow.compile()
log.info("✅ LangGraph compiled")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="DataMind AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0",
            "groq": bool(GROQ_API_KEY), "e2b": bool(E2B_API_KEY)}

@app.post("/agent/invoke")
async def invoke(request: Request):
    """
    Accepts:  { "input": { ...State fields... } }
    Returns:  { "output": { ...State fields... } }
    Exact same contract as the old langserve endpoint — frontend unchanged.
    """
    try:
        body       = await request.json()
        input_data = body.get("input", {})

        # Ensure chat_history is always a list (frontend may omit it on first call)
        if "chat_history" not in input_data:
            input_data["chat_history"] = []

        log.info(f"📩 query='{input_data.get('user_query','')[:60]}' | history={len(input_data['chat_history'])//2} turns")
        result = graph.invoke(input_data)
        return {"output": result}

    except Exception as e:
        log.error(f"❌ /agent/invoke error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info(f"🚀 Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
if __name__ == "__main__":
    log.info("🚀 Starting on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
