# Reflexion Agent

An AI-powered **Reflexion Agent** built with Python, LangGraph, Groq, and Tavily Search. The agent improves its responses through an iterative process of reflection, research, revision, and evaluation.

Unlike a basic LLM application that generates an answer in a single call, this agent uses feedback and external information retrieval to refine its output.

## Architecture

The agent follows this workflow:

```text
User Question
     |
     v
  Responder
     |
     v
  Reflection
     |
     v
   Research
     |
     v
   Revisor
     |
     v
  Evaluator
     |
     v
 Is the answer satisfactory?
     |
   YES | NO
    |    |
    v    v
   END  Reflection
         (retry)
```

## Agent Components

| Component | Responsibility |
|---|---|
| **Responder** | Generates the initial answer to the user's question. |
| **Reflection** | Critically analyzes the answer and identifies weaknesses. |
| **Research** | Generates relevant search queries and retrieves external information using Tavily Search. |
| **Revisor** | Improves the answer using the reflection and research results. |
| **Evaluator** | Evaluates the revised answer and decides whether another iteration is needed. |

## Technologies Used

- **Python** – Core programming language.
- **LangGraph** – Builds and manages the agent workflow.
- **LangChain** – LLM integration and tool interaction.
- **Groq** – Provides LLM inference.
- **Tavily Search** – Retrieves external information for research.
- **python-dotenv** – Manages environment variables.

## Key Concepts Implemented

- LLM-based agents
- Shared state management using `TypedDict`
- Nodes and edges in LangGraph
- Conditional routing
- Tool integration
- Reflection and self-critique
- Iterative answer refinement
- Controlled execution using a maximum iteration limit

## Project Structure

```text
Reflexion-Agent/
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/MoazzamKhan19/Reflexion_Agent.git
cd Reflexion_Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Add your own API keys. Never commit the `.env` file to GitHub.

### 5. Run the agent

```bash
python main.py
```

## How It Works

1. The user submits a question.
2. The Responder generates an initial response.
3. The Reflection node analyzes the response and identifies areas for improvement.
4. The Research node searches for relevant external information.
5. The Revisor generates an improved response.
6. The Evaluator checks whether the revised response meets the required quality.
7. Based on the evaluation, the graph either ends or loops back for another improvement cycle.

The workflow uses LangGraph's shared state to pass information between nodes and conditional edges to control execution.

## Future Improvements

- Improve evaluation reliability.
- Add more robust error handling.
- Optimize token usage and API calls.
- Add support for additional research tools.
- Improve the agent's stopping criteria.

## Author

**Moazzam Khan**

BS Software Engineering Student

GitHub: [MoazzamKhan19](https://github.com/MoazzamKhan19)

---

*Built as a learning project to explore Agentic AI, LangGraph workflows, and iterative LLM-based reasoning.*