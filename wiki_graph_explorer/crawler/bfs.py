"""Breadth-first crawl of a Wikipedia article subtree.

The crawler is a thin, deterministic orchestration layer on top of
:class:`crawler.wiki_api.WikiApiClient`: it maintains a FIFO frontier, fetches
out-links for every dequeued article and stops at the requested depth, at the
``max_children`` fan-out cap, or at the node budget.

References
----------
.. [1] Newman, M. E. J. (2005). *Networks: An Introduction*.
       Oxford University Press.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import config
from crawler.wiki_api import ArticleRef, WikiApiClient, WikiApiError

LOGGER = logging.getLogger("wge.crawler.bfs")

ProgressCallback = Callable[["CrawlProgress"], None]
"""Callback invoked after every processed article for the Streamlit status line."""


@dataclass
class CrawlProgress:
    """Snapshot of the crawl state used to drive the progress bar.

    Attributes
    ----------
    fetched
        Number of articles whose link list has already been retrieved.
    queued
        Number of articles still waiting in the BFS frontier.
    current
        Title of the article currently being processed (may be empty).
    total
        Number of articles discovered so far.
    """

    fetched: int
    queued: int
    current: str
    total: int

    def as_text(self) -> str:
        """Human readable one-line status.

        Returns
        -------
        str
            Status string such as ``"Information retrieval - 12 pages fetched,
            3 queued"``.
        """
        prefix = f"{self.current} - " if self.current else ""
        return f"{prefix}{self.fetched} pages fetched, {self.queued} queued"


@dataclass
class CrawlResult:
    """Immutable-ish output of a BFS crawl.

    Attributes
    ----------
    ref
        Root article reference (language + canonical title).
    links
        Mapping ``article title -> sorted list of kept out-links``.  Only
        articles that were actually fetched appear as keys.
    depths
        BFS depth of every discovered article (``0`` for the root).
    errors
        Human readable failure messages keyed by article title.
    truncated
        ``True`` when the node budget stopped the crawl before completion.
    max_children
        Fan-out cap that was applied.
    depth
        Maximum BFS depth that was requested.
    """

    ref: ArticleRef
    links: dict[str, list[str]] = field(default_factory=dict)
    depths: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    truncated: bool = False
    max_children: int = config.DEFAULT_MAX_CHILDREN
    depth: int = config.DEFAULT_DEPTH

    @property
    def root(self) -> str:
        """Canonical title of the root article.

        Returns
        -------
        str
            The root title.
        """
        return self.ref.title

    @property
    def lang(self) -> str:
        """Wikipedia language edition of the crawl.

        Returns
        -------
        str
            Two- or three-letter language code.
        """
        return self.ref.lang

    @property
    def node_count(self) -> int:
        """Number of discovered articles.

        Returns
        -------
        int
            Size of the node set.
        """
        return len(self.depths)

    @property
    def edge_count(self) -> int:
        """Number of collected directed links.

        Returns
        -------
        int
            Total number of out-edges.
        """
        return sum(len(children) for children in self.links.values())

    def article_url(self, title: str) -> str:
        """Return the browser URL of ``title`` within the crawled language.

        Parameters
        ----------
        title
            Article title.

        Returns
        -------
        str
            Percent-encoded Wikipedia URL.
        """
        from crawler.wiki_api import article_url

        return article_url(self.lang, title)


def node_budget_for_depth(depth: int) -> int:
    """Return the node budget for a BFS depth.

    Parameters
    ----------
    depth
        Requested BFS depth.

    Returns
    -------
    int
        Maximum number of articles kept in the graph.

    Examples
    --------
    >>> node_budget_for_depth(2)
    200
    >>> node_budget_for_depth(99)
    1000
    """
    return config.NODE_BUDGET_BY_DEPTH.get(depth, 1000)


def build_crawl(
    client: WikiApiClient,
    root_title: str,
    depth: int = config.DEFAULT_DEPTH,
    max_children: int = config.DEFAULT_MAX_CHILDREN,
    progress: ProgressCallback | None = None,
    node_budget: int | None = None,
) -> CrawlResult:
    """Crawl the article subtree rooted at ``root_title``.

    The traversal is a plain FIFO BFS: articles are dequeued in insertion
    order, their out-links are sorted alphabetically and truncated to
    ``max_children`` items, and the crawl stops when the frontier contains only
    articles deeper than ``depth``.  Given identical inputs the traversal
    visits exactly the same articles in exactly the same order, which is what
    makes all downstream centrality values reproducible [1]_.

    Parameters
    ----------
    client
        Configured MediaWiki client for the target language edition.
    root_title
        Title of the root article (already resolved, see
        :meth:`WikiApiClient.resolve_title`).
    depth
        Maximum BFS depth (``1`` = the root and its direct links).
    max_children
        Maximum number of out-links kept per article.
    progress
        Optional callback receiving a :class:`CrawlProgress` after each fetch.
    node_budget
        Hard cap on the number of nodes; defaults to the value configured for
        ``depth`` in :data:`config.NODE_BUDGET_BY_DEPTH`.

    Returns
    -------
    CrawlResult
        The crawled subtree, including non-fatal per-article errors.

    Examples
    --------
    >>> result = build_crawl(client, "Information retrieval", 1, 3)  # doctest: +SKIP
    >>> result.node_count                                           # doctest: +SKIP
    4
    """
    depth = max(1, min(int(depth), config.MAX_DEPTH))
    max_children = max(1, int(max_children))
    budget = (
        int(node_budget) if node_budget is not None else node_budget_for_depth(depth)
    )
    result = CrawlResult(
        ref=ArticleRef(lang=client.lang, title=root_title),
        max_children=max_children,
        depth=depth,
        depths={root_title: 0},
    )
    frontier: deque[str] = deque([root_title])
    fetched = 0

    while frontier:
        current = frontier.popleft()
        if result.depths[current] >= depth:
            continue
        if progress is not None:
            progress(
                CrawlProgress(
                    fetched=fetched,
                    queued=len(frontier),
                    current=current,
                    total=result.node_count,
                )
            )
        try:
            children = client.fetch_links(current, max_children=max_children)
        except WikiApiError as error:
            LOGGER.warning("Skipping '%s': %s", current, error)
            result.errors[current] = str(error)
            continue
        result.links[current] = children
        fetched += 1
        for child in children:
            if child not in result.depths:
                if len(result.depths) >= budget:
                    result.truncated = True
                    LOGGER.warning("Node budget of %d reached; crawl truncated", budget)
                    break
                result.depths[child] = result.depths[current] + 1
                frontier.append(child)

    if progress is not None:
        progress(
            CrawlProgress(
                fetched=fetched,
                queued=0,
                current="",
                total=result.node_count,
            )
        )
    LOGGER.info(
        "Crawl finished: %d nodes, %d edges, %d errors",
        result.node_count,
        result.edge_count,
        len(result.errors),
    )
    return result
