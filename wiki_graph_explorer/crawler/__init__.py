"""Wikipedia web-graph crawler (MediaWiki API client + BFS traversal)."""

from __future__ import annotations

from .bfs import CrawlProgress, CrawlResult, build_crawl, node_budget_for_depth
from .wiki_api import (
    ArticleRef,
    WikiApiClient,
    WikiApiError,
    article_url,
    normalise_title,
    parse_article_input,
)

__all__ = [
    "ArticleRef",
    "CrawlProgress",
    "CrawlResult",
    "WikiApiClient",
    "WikiApiError",
    "article_url",
    "build_crawl",
    "node_budget_for_depth",
    "normalise_title",
    "parse_article_input",
]
