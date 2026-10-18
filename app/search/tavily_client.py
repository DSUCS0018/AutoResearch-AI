"""
app/search/tavily_client.py
---------------------------
Reusable Tavily search client for AutoResearch AI.

All Tavily SDK interactions are isolated here. The rest of the application
imports TavilySearchClient and calls search() without knowing API details.

Usage:
    from app.search.tavily_client import TavilySearchClient

    client = TavilySearchClient()
    sources = client.search("latest developments in artificial intelligence")
    for source in sources:
        print(source.title, source.url, source.score)
"""

import logging
from typing import Optional

from tavily import TavilyClient
from tavily.errors import MissingAPIKeyError, InvalidAPIKeyError, UsageLimitExceededError

from app.config import config
from app.search.models import Source

logger = logging.getLogger(__name__)


class SearchConfigurationError(Exception):
    """Raised when the search client cannot be configured (e.g. missing key)."""


class SearchAPIError(Exception):
    """Raised when the search API call fails."""


class TavilySearchClient:
    """
    Thin wrapper around the official Tavily Python SDK.

    Responsibilities:
      - Validate configuration on construction.
      - Execute searches and return structured Source objects.
      - Translate SDK errors into clean application-level errors.
      - Never log the API key.
    """

    def __init__(self) -> None:
        """
        Initialize the Tavily client.

        Raises:
            SearchConfigurationError: If TAVILY_API_KEY is not configured.
        """
        if not config.TAVILY_API_KEY:
            raise SearchConfigurationError(
                "TAVILY_API_KEY is not configured. "
                "Please add it to your .env file."
            )

        # Key is passed explicitly — never exposed in logs.
        self._client = TavilyClient(api_key=config.TAVILY_API_KEY)
        self._default_max_results = config.TAVILY_MAX_RESULTS
        logger.debug(
            "TavilySearchClient initialised (default_max_results=%d)",
            self._default_max_results,
        )

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> list[Source]:
        """
        Search the web for a given query using the Tavily API.

        Args:
            query:       The search query string.
            max_results: Number of results to return. Falls back to
                         TAVILY_MAX_RESULTS from config if not specified.

        Returns:
            A list of Source objects ordered by relevance score.

        Raises:
            ValueError:                 If query is blank.
            SearchConfigurationError:   If the API key is rejected.
            SearchAPIError:             On API or network failures.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        limit = max_results if max_results is not None else self._default_max_results
        logger.info("Searching Tavily for: %r (max_results=%d)", query, limit)

        try:
            response = self._client.search(
                query=query,
                max_results=limit,
                search_depth="basic",
            )
        except (MissingAPIKeyError, InvalidAPIKeyError) as exc:
            raise SearchConfigurationError(
                "Tavily authentication failed. "
                "Verify that TAVILY_API_KEY in your .env file is correct."
            ) from exc
        except UsageLimitExceededError as exc:
            raise SearchAPIError(
                "Tavily usage limit exceeded. "
                "Check your plan at app.tavily.com."
            ) from exc
        except Exception as exc:
            raise SearchAPIError(
                f"Tavily search failed: {type(exc).__name__}: {exc}"
            ) from exc

        sources = self._parse_results(response)
        logger.info("Tavily returned %d results.", len(sources))
        return sources

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(response: dict) -> list[Source]:
        """
        Convert the Tavily API response dict into a list of Source objects.

        The Tavily SDK returns a dict with a 'results' key containing a list
        of result dicts, each with 'title', 'url', 'content', and 'score'.
        """
        results = response.get("results", [])
        sources: list[Source] = []

        for item in results:
            sources.append(
                Source(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                )
            )

        return sources
