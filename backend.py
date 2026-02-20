import os
from e2b_code_interpreter import Sandbox
from typing import Optional
from dotenv import load_dotenv

from pathlib import Path

# This ALWAYS finds .env in the same folder as backend.py
# regardless of where your terminal is running from


import base64
import io
import pandas as pd
from typing import TypedDict

# FastAPI Components
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

# LangGraph & AI Components
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Environment Variables
os.environ["GROQ_API_KEY"] = "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"
os.environ["E2B_API_KEY"] = "e2b_cfdff01f4756bea245d24afcebde1b06fb50651d"

# 2. Define State
class State(TypedDict):
    user_query: str
    file_data: str
    extracted_data: str
    code_generated_llm: str
    code_output: Optional[str]      # ✅ Optional to allow None
    image_data: Optional[str]       # ✅ Optional to allow None
    code_error: Optional[str]

# 3. Define Nodes

def DataExtractor(state: State):
    print("--- NODE 1: Extracting Data ---")
    file_b64 = state['file_data']
    
    try:
        file_bytes = base64.b64decode(file_b64)
        df = pd.read_csv(io.BytesIO(file_bytes))
        cols = df.columns.tolist()
        return {"extracted_data": f"Columns: {cols}\n{df.head().to_string()}"}
    except Exception as e:
        return {"code_error": f"Data Extraction Error: {str(e)}"}

def Code_generator(state: State):
    print("--- NODE 2: Generating Code ---")
    query = state['user_query']
    data_preview = state['extracted_data']
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python Data Analyst.
        - The dataset is at: '/app/data.csv'
        - Load it: df = pd.read_csv('/app/data.csv')
        - Print the final answer.
        - If plotting, use plt.show(). and one more thing the graph should be stylished  with beautiful colors and themes.
        - Return ONLY python code in backticks.
        """),
        ("human", "Query: {query}\n\nData Info:\n{data_preview}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({"query": query, "data_preview": data_preview})
    clean_code = res.content.replace("```python", "").replace("```", "").strip()
    
    return {"code_generated_llm": clean_code}

def CodeExecutorNode(state: State):
    print("--- NODE 3: Executing in E2B ---")
    code = state['code_generated_llm']
    file_b64 = state['file_data']
    
    if not code:
        return {
            "code_output": "",
            "image_data": "",
            "code_error": "No code generated"
        }

    try:
        print("Creating sandbox...")
        
        with Sandbox.create(timeout=300) as sandbox:
            print("Sandbox created successfully!")
            
            # 1. Upload File
            if file_b64:
                print("Uploading file...")
                file_bytes = base64.b64decode(file_b64)
                sandbox.files.write("/app/data.csv", file_bytes)
                print("File uploaded!")
            
            # 2. Run Code
            print("Running code...")
            execution = sandbox.run_code(code, timeout=300)
            
            # 3. Check for Runtime Errors
            if execution.error:
                return {
                    "code_output": "",
                    "image_data": "",
                    "code_error": f"Runtime Error: {execution.error.name}: {execution.error.value}"
                }
            
            # 4. Get Outputs
            img = ""                                          # ✅ empty string not None
            if execution.results:
                for res in execution.results:
                    if hasattr(res, 'png') and res.png:
                        img = res.png
                        break

            # ✅ Fix — stdout can be a list, join into single string
            if execution.logs and execution.logs.stdout:
                logs = execution.logs.stdout
                output = "\n".join(logs) if isinstance(logs, list) else str(logs)
            else:
                output = ""                                   # ✅ empty string not None
            
            return {
                "code_output": output,                        # ✅ always a string
                "image_data": img,                            # ✅ always a string
                "code_error": ""                              # ✅ empty string not None
            }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Sandbox Error: {error_msg}")
        return {
            "code_output": "",                                # ✅ always a string
            "image_data": "",                                 # ✅ always a string
            "code_error": f"Sandbox Error: {error_msg}"
        }

# 4. Build Graph
workflow = StateGraph(State)
workflow.add_node("DataExtractor", DataExtractor)
workflow.add_node("CodeGenerator", Code_generator)
workflow.add_node("CodeExecutor", CodeExecutorNode)

workflow.add_edge(START, "DataExtractor")
workflow.add_edge("DataExtractor", "CodeGenerator")
workflow.add_edge("CodeGenerator", "CodeExecutor")
workflow.add_edge("CodeExecutor", END)

graph = workflow.compile()

# 5. Create FastAPI App
app = FastAPI(title="AI Data Analyst Backend")

add_routes(app, graph, path="/agent")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
