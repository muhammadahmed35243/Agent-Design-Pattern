import os
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

def search(query: str) -> str:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not found in environment."

    params = {
        "q": query,
        "location": "Pakistan",
        "hl": "en",
        "gl": "us",
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            return f"SerpAPI error: {results['error']}"

        if "organic_results" in results and len(results["organic_results"]) > 0:
            top = results["organic_results"][0]
            title = top.get("title", "No Title")
            snippet = top.get("snippet", "No Snippet")
            link = top.get("link", "No Link")
            return f"🔍 **{title}**\n\n{snippet}\n\n🔗 [Read more]({link})"
        else:
            return "No search results found."

    except Exception as e:
        return f"Exception during search: {str(e)}"
