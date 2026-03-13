DataMind - AI Data Analyst Agent
An intelligent AI-powered data analysis agent that automatically generates and executes Python code to analyze your CSV files using LangGraph, Groq LLM, and E2B Sandbox.

Features
Intelligent Analysis: Ask natural language questions about your data

AI-Powered: Uses Groq's Llama 3.3 70B model for code generation

Real-time Execution: Executes Python code in isolated E2B sandboxes

Data Visualization: Automatically generates plots and visualizations

Modern Stack: Built with LangGraph, FastAPI, and Streamlit

Multi-step Workflow: Extract → Generate → Execute → Visualize

How It Works
Data Extraction: Reads and previews your CSV file.

Code Generation: AI generates Python code to answer your question.

Sandbox Execution: Runs code safely in E2B isolated environment.

Visualization: Displays results and generated plots.

Installation
Clone the Repository:
git clone https://github.com/EngNumanAli/datamind-analyst.git

Create Virtual Environment:
python -m venv datamind

Install Dependencies:
pip install -r requirements.txt

Configuration
You need to create a .env file in the project root with your API keys. Do not share these keys publicly.

GROQ_API_KEY=your_key_here

E2B_API_KEY=your_key_here

Usage
Start Backend: Run python backend.py.

Start Frontend: Run streamlit run frontend.py.

Upload: Drop your CSV into the browser and ask a question.

Project Structure
backend.py: FastAPI backend with LangGraph logic.

frontend.py: Streamlit user interface.

data/: Directory for your CSV files.

docs/: Detailed manuals and troubleshooting.

Security & Performance
Code is executed in isolated E2B sandboxes to protect your host machine.

Typical analysis takes 10-30 seconds depending on complexity.

Recommended file size is up to 10MB.DataMind - AI Data Analyst Agent
An intelligent AI-powered data analysis agent that automatically generates and executes Python code to analyze your CSV files using LangGraph, Groq LLM, and E2B Sandbox.

Features
Intelligent Analysis: Ask natural language questions about your data

AI-Powered: Uses Groq's Llama 3.3 70B model for code generation

Real-time Execution: Executes Python code in isolated E2B sandboxes

Data Visualization: Automatically generates plots and visualizations

Modern Stack: Built with LangGraph, FastAPI, and Streamlit

Multi-step Workflow: Extract → Generate → Execute → Visualize

How It Works
Data Extraction: Reads and previews your CSV file.

Code Generation: AI generates Python code to answer your question.

Sandbox Execution: Runs code safely in E2B isolated environment.

Visualization: Displays results and generated plots.

Installation
Clone the Repository:
git clone https://github.com/yourusername/datamind-analyst.git

Create Virtual Environment:
python -m venv datamind

Install Dependencies:
pip install -r requirements.txt

Configuration
You need to create a .env file in the project root with your API keys. Do not share these keys publicly.

GROQ_API_KEY=your_key_here

E2B_API_KEY=your_key_here

Usage
Start Backend: Run python backend.py.

Start Frontend: Run streamlit run frontend.py.

Upload: Drop your CSV into the browser and ask a question.

Project Structure
backend.py: FastAPI backend with LangGraph logic.

frontend.py: Streamlit user interface.

data/: Directory for your CSV files.

docs/: Detailed manuals and troubleshooting.

Security & Performance
Code is executed in isolated E2B sandboxes to protect your host machine.

Typical analysis takes 10-30 seconds depending on complexity.

Recommended file size is up to 10MB.
