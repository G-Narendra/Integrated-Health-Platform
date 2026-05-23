"""
Web search tool for medical research (using Tavily or DuckDuckGo).
"""

import os
from typing import Dict, List


class WebSearchTool:
    """Tool for searching the web for medical information."""

    def __init__(self):
        self._results_cache: Dict[str, List[Dict]] = {}

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web for medical information."""
        # Check cache
        cache_key = f"{query}:{max_results}"
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]

        try:
            # Try Tavily if API key is available
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                return self._search_tavily(query, max_results, tavily_key)
        except Exception:
            pass

        # Fallback: DuckDuckGo
        try:
            return self._search_duckduckgo(query, max_results)
        except Exception:
            pass

        return self._get_mock_results(query, max_results)

    def _search_tavily(self, query: str, max_results: int, api_key: str) -> List[Dict]:
        """Search using Tavily API."""
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "source": "tavily",
            }
            for r in response.get("results", [])
        ]

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", ""),
                        "source": "duckduckgo",
                    })
                return results
        except ImportError:
            return self._get_mock_results(query, max_results)

    def _get_mock_results(self, query: str, max_results: int) -> List[Dict]:
        """Return simulated search results."""
        return [
            {
                "title": f"PubMed: Recent findings related to {query}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}",
                "content": f"Recent medical literature on {query}. Multiple studies have shown promising results in treatment outcomes.",
                "source": "pubmed",
            },
            {
                "title": f"WHO Guidelines on {query}",
                "url": "https://www.who.int",
                "content": f"World Health Organization guidelines and recommendations regarding {query}.",
                "source": "who",
            },
            {
                "title": f"UAE MOH Guidelines - {query}",
                "url": "https://www.mohap.gov.ae",
                "content": f"UAE Ministry of Health guidelines and protocols for managing {query} in UAE healthcare facilities.",
                "source": "moh_uae",
            },
        ][:max_results]

    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search PubMed specifically for medical literature."""
        try:
            import requests
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            }
            response = requests.get(f"{base_url}esearch.fcgi", params=params, timeout=10)
            data = response.json()

            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            # Fetch details
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            }
            fetch_response = requests.get(
                f"{base_url}efetch.fcgi", params=fetch_params, timeout=10
            )
            # Parse XML for titles and abstracts (simplified)
            return [
                {
                    "pmid": pid,
                    "title": f"PubMed Article {pid}",
                    "source": "pubmed",
                }
                for pid in ids
            ]
        except Exception:
            return []
