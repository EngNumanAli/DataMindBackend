import os
import logging
import base64
import io
import time
from pathlib import Path
from typing import Optional, TypedDict, List
from functools import wraps

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Validate API Keys ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
E2B_API_KEY  = os.getenv("E2B_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set. Check your .env file.")
if not E2B_API_KEY:
    raise EnvironmentError("E2B_API_KEY is not set. Check your .env file.")

log.info("✅ API keys loaded successfully")

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langserve import add_routes
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from e2b_code_interpreter import Sandbox
import uvicorn

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB      = 10
MAX_FILE_SIZE_BYTES   = MAX_FILE_SIZE_MB * 1024 * 1024
FILE_PATH_IN_SANDBOX  = "/app/data.csv"
LLM_PREVIEW_MAX_CHARS = 1500
LLM_PREVIEW_MAX_ROWS  = 3
LLM_PREVIEW_MAX_COLS  = 15
MAX_HISTORY_TURNS     = 6  # keep last 6 exchanges to avoid token overflow

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    user_query:          str
    file_data:           str
    extracted_data:      str
    code_generated_llm:  str
    code_output:         Optional[str]
    image_data:          Optional[str]
    code_error:          Optional[str]
    # NEW — conversation history as list of dicts {role, content}
    chat_history:        List[dict]

# ── Helpers ───────────────────────────────────────────────────────────────────
def retry(max_attempts=3, delay=2.0, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    log.warning(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

def safe_str(value) -> str:
    if value is None: return ""
    if isinstance(value, list): return "\n".join(str(i) for i in value)
    if isinstance(value, bytes): return value.decode("utf-8", errors="replace")
    return str(value)

def history_to_messages(history: list) -> list:
    """
    Converts chat history list of dicts into LangChain message objects.
    Each item is {role: 'human'|'ai', content: '...'}
    Only keeps the last MAX_HISTORY_TURNS pairs to stay within token limits.
    """
    messages = []
    # Trim to last N turns (each turn = 1 human + 1 ai message)
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    for msg in trimmed:
        role    = msg.get("role", "human")
        content = msg.get("content", "")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages

# ── NODE 1: Data Extractor ────────────────────────────────────────────────────
def DataExtractor(state: State):
    log.info("NODE 1 | DataExtractor | Starting")
    file_b64 = state["file_data"]

    if not file_b64:
        return {"code_error": "No file uploaded. Please upload a CSV file."}

    try:
        file_bytes = base64.b64decode(file_b64)
        size_mb = len(file_bytes) / (1024 * 1024)
        log.info(f"NODE 1 | File size: {size_mb:.2f} MB")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return {"code_error": f"File too large ({size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB."}

        encodings = ["utf-8", "latin-1", "windows-1252", "iso-8859-1"]
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                log.info(f"NODE 1 | Encoding: {enc}")
                break
            except Exception:
                continue

        if df is None:
            return {"code_error": "Could not read the CSV file. Please check the format."}

        cols       = df.columns.tolist()
        preview_df = df.head(LLM_PREVIEW_MAX_ROWS)
        col_note   = ""

        if len(cols) > LLM_PREVIEW_MAX_COLS:
            preview_df = preview_df.iloc[:, :LLM_PREVIEW_MAX_COLS]
            col_note   = f"\n(showing {LLM_PREVIEW_MAX_COLS} of {len(cols)} columns)"

        preview_str = preview_df.to_string()
        if len(preview_str) > LLM_PREVIEW_MAX_CHARS:
            preview_str = preview_str[:LLM_PREVIEW_MAX_CHARS] + "\n...(truncated)"

        extracted = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\nColumns: {cols}{col_note}\n\nSample:\n{preview_str}"
        log.info("NODE 1 | Complete")
        return {"extracted_data": extracted}

    except Exception as e:
        log.error(f"NODE 1 | Failed: {e}")
        return {"code_error": f"Data Extraction Error: {str(e)}"}


# ── NODE 2: Code Generator (with memory) ─────────────────────────────────────
@retry(max_attempts=3, delay=3.0)
def _call_llm_with_history(llm, system_msg: str, history_messages: list, human_msg: str) -> str:
    """
    Calls the LLM with the full conversation history so it remembers
    previous queries and their results in the same session.
    """
    messages = [SystemMessage(content=system_msg)]
    messages.extend(history_messages)  # previous exchanges
    messages.append(HumanMessage(content=human_msg))
    response = llm.invoke(messages)
    return response.content


def Code_generator(state: State):
    log.info("NODE 2 | CodeGenerator | Starting")
    query        = state["user_query"]
    data_preview = state["extracted_data"]
    history      = state.get("chat_history", [])

    if not query or not query.strip():
        return {"code_error": "No query provided."}
    if not data_preview:
        return {"code_error": "No data preview. Data extraction may have failed."}

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=GROQ_API_KEY
        )

        system_msg = """You are an expert Python Data Analyst with memory of our conversation.

Rules:
- The dataset is ALWAYS at: '/app/data.csv'
- ALWAYS load with: df = pd.read_csv('/app/data.csv', encoding='latin-1')
- Always print your final answer clearly.
- If plotting: use matplotlib dark theme. Use plt.style.use('dark_background'). Vibrant colors. Call plt.tight_layout() before plt.show().
- Import all libraries at the top.
- You remember previous questions and results in this session — use that context.
- Return ONLY raw Python code. No explanations. No markdown. No backticks."""

        # Build the human message including data context
        human_msg = f"Data Info:\n{data_preview}\n\nQuery: {query}"

        # Convert history to LangChain messages
        history_messages = history_to_messages(history)
        log.info(f"NODE 2 | History turns: {len(history_messages)}")

        raw_content = _call_llm_with_history(llm, system_msg, history_messages, human_msg)

        clean_code = (
            raw_content
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

        log.info(f"NODE 2 | Code generated ({len(clean_code)} chars)")
        return {"code_generated_llm": clean_code}

    except Exception as e:
        log.error(f"NODE 2 | Failed: {e}")
        error_msg = str(e)
        if "413" in error_msg or "too large" in error_msg.lower():
            return {"code_error": "Dataset preview too large for AI. Try a smaller file."}
        if "rate_limit" in error_msg.lower():
            return {"code_error": "Rate limit reached. Wait 30 seconds and try again."}
        return {"code_error": f"Code Generation Error: {error_msg}"}


# ── NODE 3: Code Executor ─────────────────────────────────────────────────────
def CodeExecutorNode(state: State):
    log.info("NODE 3 | CodeExecutor | Starting")
    code     = state["code_generated_llm"]
    file_b64 = state["file_data"]
    query    = state["user_query"]

    if not code or not code.strip():
        return {"code_output": "", "image_data": "", "code_error": "No code was generated.", "chat_history": state.get("chat_history", [])}

    code = code.replace("'data.csv'",     f"'{FILE_PATH_IN_SANDBOX}'")
    code = code.replace('"data.csv"',     f'"{FILE_PATH_IN_SANDBOX}"')
    code = code.replace("/home/data.csv", FILE_PATH_IN_SANDBOX)
    code = code.replace("/tmp/data.csv",  FILE_PATH_IN_SANDBOX)

    try:
        log.info("NODE 3 | Creating sandbox")
        with Sandbox.create(timeout=300) as sandbox:
            log.info("NODE 3 | Sandbox ready")

            if file_b64:
                file_bytes = base64.b64decode(file_b64)
                sandbox.files.write(FILE_PATH_IN_SANDBOX, file_bytes)

                verify = sandbox.run_code(f"""
import os
exists = os.path.exists('{FILE_PATH_IN_SANDBOX}')
size   = os.path.getsize('{FILE_PATH_IN_SANDBOX}') if exists else 0
print(f"File exists: {{exists}}, Size: {{size}} bytes")
""")
                log.info(f"NODE 3 | File verify: {safe_str(verify.logs.stdout)}")

            execution = sandbox.run_code(code, timeout=120)

            if execution.error:
                error_name  = getattr(execution.error, "name",  "Error")
                error_value = getattr(execution.error, "value", str(execution.error))
                log.error(f"NODE 3 | Runtime error: {error_name}")

                human_error = f"Runtime Error: {error_name}: {error_value}"
                if "ModuleNotFoundError" in error_name:
                    human_error = f"Required library not installed: {error_value}"
                elif "FileNotFoundError" in error_name:
                    human_error = "Data file not found. Please try again."
                elif "MemoryError" in error_name:
                    human_error = "Dataset too large. Try a smaller file."

                # Still update history with the failed attempt
                updated_history = _update_history(
                    state.get("chat_history", []),
                    query,
                    f"[Error] {human_error}"
                )
                return {"code_output": "", "image_data": "", "code_error": human_error, "chat_history": updated_history}

            img = ""
            if execution.results:
                for res in execution.results:
                    if hasattr(res, "png") and res.png:
                        img = res.png
                        log.info("NODE 3 | Chart captured")
                        break

            output = ""
            if execution.logs and execution.logs.stdout:
                output = safe_str(execution.logs.stdout)

            log.info(f"NODE 3 | Done | output: {len(output)} chars | image: {bool(img)}")

            # ── Update conversation history ───────────────────────────────
            # Store query + result summary so LLM remembers in next turn
            ai_response = output if output else "[Code executed successfully"
            if img:
                ai_response += " — chart generated]"
            else:
                ai_response += "]"

            updated_history = _update_history(
                state.get("chat_history", []),
                query,
                ai_response
            )

            return {
                "code_output":  output,
                "image_data":   img,
                "code_error":   "",
                "chat_history": updated_history   # ← history grows each turn
            }

    except Exception as e:
        error_msg = str(e)
        log.error(f"NODE 3 | Sandbox error: {error_msg}")

        if "timed out" in error_msg.lower():
            human = "Execution timed out. Try a simpler query."
        elif "getaddrinfo" in error_msg or "11002" in error_msg:
            human = "Cannot reach E2B servers. Check your internet connection."
        else:
            human = f"Sandbox Error: {error_msg}"

        updated_history = _update_history(state.get("chat_history", []), query, f"[Error] {human}")
        return {"code_output": "", "image_data": "", "code_error": human, "chat_history": updated_history}


def _update_history(history: list, user_msg: str, ai_msg: str) -> list:
    """
    Appends the latest exchange to history.
    Keeps history trimmed to MAX_HISTORY_TURNS pairs.
    """
    updated = list(history)
    updated.append({"role": "human", "content": user_msg})
    updated.append({"role": "ai",    "content": ai_msg})
    # Trim — keep last N pairs (each pair = 2 items)
    max_items = MAX_HISTORY_TURNS * 2
    if len(updated) > max_items:
        updated = updated[-max_items:]
    return updated


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
log.info("✅ LangGraph pipeline compiled")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Data Analyst API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status":          "healthy",
        "version":         "1.0.0",
        "groq_key_loaded": bool(GROQ_API_KEY),
        "e2b_key_loaded":  bool(E2B_API_KEY),
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

add_routes(app, graph, path="/agent")
log.info("✅ Routes added at /agent")

if __name__ == "__main__":
    log.info("🚀 Starting on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
