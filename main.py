
# ==================================================
# 1. IMPORTS AND CONFIGURATION
# ==================================================

import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=3000,
    reasoning_effort="low"
)

search_tool = TavilySearch(max_results=3)

MAX_ITERATIONS = 2


# ==================================================
# 2. LANGGRAPH STATE
# ==================================================

class AgentState(TypedDict):
    question: str
    initial_answer: str
    current_answer: str
    critique: str
    search_queries: List[str]
    search_results: list
    final_answer: str
    iteration: int
    is_satisfactory: bool


# ==================================================
# 3. RESPONDER NODE
# ==================================================

def responder_node(state: AgentState):

    print("\n========== RESPONDER ==========")

    response = llm.invoke(state["question"])

    initial_answer = response.content

    print("\nInitial Answer:")
    print(initial_answer)

    return {
        "initial_answer": initial_answer,
        "current_answer": initial_answer
    }


# ==================================================
# 4. REFLECTION NODE
# ==================================================

def reflection_node(state: AgentState):

    print("\n========== REFLECTION ==========")

    reflection_prompt = f"""
    You are a critical reviewer.

    Review this answer to the question.

    Question:
    {state["question"]}

    Answer:
    {state["current_answer"]}

    Identify:
    1. Incorrect factual claims.
    2. Unsupported information.
    3. Important missing details.

    Give a concise critique in no more than 200 words.
    Use bullet points. Do not rewrite the answer.
    """

    reflection = llm.invoke(reflection_prompt)

    critique = reflection.content

    print("\nCritique:")
    print(critique)

    return {
        "critique": critique
    }


# ==================================================
# 5. RESEARCH NODE - QUERY GENERATION + TAVILY
# ==================================================

def research_node(state: AgentState):

    print("\n========== RESEARCH ==========")

    search_query_prompt = f"""
    You are a research assistant.

    Based on the following critique, generate 3 short web search
    queries that can help verify the claims.

    Original Question:
    {state["question"]}

    Current Answer:
    {state["current_answer"]}

    Reflection:
    {state["critique"]}

    Return only 3 search queries, one per line.
    """

    query_response = llm.invoke(search_query_prompt)

    search_queries = query_response.content

    queries = [
        query.strip("- ").strip()
        for query in search_queries.splitlines()
        if query.strip()
    ]

    all_search_results = []

    for query in queries:

        print("\nSearching:", query)

        results = search_tool.invoke({
            "query": query
        })

        for result in results.get("results", []):

            search_data = {
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content")
            }

            all_search_results.append(search_data)

    print("\nTavily Results Retrieved:", len(all_search_results))

    return {
        "search_queries": queries,
        "search_results": all_search_results
    }


# ==================================================
# 6. REVISOR NODE
# ==================================================

def revisor_node(state: AgentState):

    print("\n========== REVISOR ==========")

    revisor_prompt = f"""
    You are an expert answer reviser.

    Improve the current answer using the reflection
    and external search evidence.

    Original Question:
    {state["question"]}

    Initial Answer:
    {state["initial_answer"]}

    Current Answer:
    {state["current_answer"]}

    Reflection / Critique:
    {state["critique"]}

    External Search Results:
    {state["search_results"]}

    Instructions:
    1. Correct factual errors identified by the reflection.
    2. Use relevant information from external search results.
    3. Preserve information that is already correct.
    4. Do not blindly trust search snippets.
    5. Mention uncertainty when evidence is insufficient.
    6. Do not invent facts.
    7. Provide a clear, accurate, well-structured answer.
    """

    revised_response = llm.invoke(revisor_prompt)

    final_answer = revised_response.content

    new_iteration = state["iteration"] + 1

    print("\nRevised Answer:")
    print(final_answer)

    print("\nRevision Number:", new_iteration)

    return {
        "current_answer": final_answer,
        "final_answer": final_answer,
        "iteration": new_iteration
    }



# ==================================================
# 7. EVALUATOR NODE
# ==================================================

def evaluator_node(state: AgentState):

    print("\n========== EVALUATOR ==========")

    evaluation_prompt = f"""
    You are an answer quality evaluator.

    Original Question:
    {state["question"]}

    Revised Answer:
    {state["current_answer"]}

    Previous Critique:
    {state["critique"]}

    Evaluate whether the revised answer:
    1. Answers the original question.
    2. Addresses the important issues in the critique.
    3. Avoids unsupported or obviously incorrect claims.

    Respond with exactly one word:
    YES if satisfactory.
    NO if further improvement is needed.
    """

    # Limit the evaluator's output tokens
    evaluator_llm = llm.bind(max_tokens=20)

    evaluation = evaluator_llm.invoke(evaluation_prompt)

    decision = evaluation.content.strip().upper()

    is_satisfactory = decision.startswith("YES")

    print("\nEvaluation:", decision)

    return {
        "is_satisfactory": is_satisfactory
    }


# ==================================================
# 8. CONDITIONAL ROUTING
# ==================================================

def should_continue(state: AgentState):

    if state["is_satisfactory"]:
        print("\nAnswer accepted. Ending graph.")
        return "end"

    if state["iteration"] >= MAX_ITERATIONS:
        print("\nMaximum iterations reached. Ending graph.")
        return "end"

    print("\nAnswer needs improvement. Repeating reflection.")
    return "continue"


# ==================================================
# 9. BUILD LANGGRAPH
# ==================================================

graph_builder = StateGraph(AgentState)

# Register nodes
graph_builder.add_node("responder", responder_node)
graph_builder.add_node("reflection", reflection_node)
graph_builder.add_node("research", research_node)
graph_builder.add_node("revisor", revisor_node)
graph_builder.add_node("evaluator", evaluator_node)

# Connect nodes
graph_builder.add_edge(START, "responder")
graph_builder.add_edge("responder", "reflection")
graph_builder.add_edge("reflection", "research")
graph_builder.add_edge("research", "revisor")
graph_builder.add_edge("revisor", "evaluator")

# Conditional edge
graph_builder.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "continue": "reflection",
        "end": END
    }
)

# Compile graph
app = graph_builder.compile()


# ==================================================
# 10. EXECUTE GRAPH
# ==================================================

question = input("\nEnter your Question: ")

result = app.invoke({
    "question": question,
    "iteration": 0
})

print("\n========== FINAL ANSWER ==========")
print(result["final_answer"])
