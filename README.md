======DataMind - AI Data Analyst Agent======

An intelligent AI-powered data analysis agent that automatically generates and executes Python code to analyze your CSV files using LangGraph, Groq LLM, and E2B Sandbox.

----Features----

1-Intelligent Analysis: Ask natural language questions about your data

2-AI-Powered: Uses Groq's Llama 3.3 70B model for code generation

3-Real-time Execution: Executes Python code in isolated E2B sandboxes

4-Data Visualization: Automatically generates plots and visualizations

----Modern Stack----

Built with LangGraph, FastAPI, and Streamlit

----Multi-step Workflow----

Extract → Generate → Execute → Visualize
----How It Works----
1-Data Extraction: Reads and previews your CSV file.

2-Code Generation: AI generates Python code to answer your question.

3-Sandbox Execution: Runs code safely in E2B isolated environment.

4-Visualization: Displays results and generated plots.

----Installation----

1-Clone the Repository:
 git clone https://github.com/EngNumanAli/datamind-analyst.git

2-Create Virtual Environment:
python -m venv datamind

3-Install Dependencies:
pip install -r requirements.txt

4-Configuration
You need to create a .env file in the project root with your API keys. Do not share these keys publicly.

5-GROQ_API_KEY=your_key_here

6-E2B_API_KEY=your_key_here

----Usage----
1-Start Backend: Run python backend.py.

2-Start Frontend: Run streamlit run frontend.py.

  Upload: Drop your CSV into the browser and ask a question.

----Project Structure----

backend.py: FastAPI backend with LangGraph logic.

frontend.py: Streamlit user interface.

data/: Directory for your CSV files.

docs/: Detailed manuals and troubleshooting.

----Security & Performance----

Code is executed in isolated E2B sandboxes to protect your host machine.

Typical analysis takes 10-30 seconds depending on complexity.

Recommended file size is up to 10MB.
