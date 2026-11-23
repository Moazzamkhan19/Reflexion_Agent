
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# Initialize Tavily search
search_tool = TavilySearch(max_results=3)

# Search the web
results = search_tool.invoke({
    "query": "What is an AI agent in artificial intelligence?"
})

# Display results
for result in results["results"]:
    print("Title:", result.get("title"))
    print("URL:", result.get("url"))
    print("Content:", result.get("content"))
    print("-" * 50)
