"""Application-wide constants for the Wiki Graph Centrality Explorer.

This module is intentionally free of side effects other than reading
environment variables, so that every other module can import it safely both
from the Streamlit app and from the unit tests.

All tunables live here (API endpoint, cache layout, crawl limits, centrality
hyper-parameters, colours and server ports) which keeps the rest of the code
free of magic numbers.

References
----------
.. [1] Brin, S., & Page, L. (1998). The anatomy of a large-scale
       hypertextual information retrieval system. *Computer Networks and
       Systems*, 30(1-7), 301-309.
.. [2] Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked
       environment. *Journal of the ACM*, 46(5), 604-632.
.. [3] Newman, M. E. J. (2005). *Networks: An Introduction*. Oxford
       University Press.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
"""Directory that contains this file; every relative path is anchored here."""

CACHE_DIR: Final[Path] = Path(os.environ.get("WGE_CACHE_DIR", str(BASE_DIR / "cache")))
"""Directory holding the on-disk JSON cache of MediaWiki responses."""

LOG_DIR: Final[Path] = Path(os.environ.get("WGE_LOG_DIR", str(BASE_DIR / "logs")))
"""Directory holding ``app.log``."""

LOG_FILE: Final[Path] = LOG_DIR / "app.log"
"""Full path of the application log file."""

# --------------------------------------------------------------------------
# MediaWiki API
# --------------------------------------------------------------------------
API_URL_TEMPLATE: Final[str] = "https://{lang}.wikipedia.org/w/api.php"
"""MediaWiki ``api.php`` endpoint for a given language edition."""

ARTICLE_URL_TEMPLATE: Final[str] = "https://{lang}.wikipedia.org/wiki/{title}"
"""Human readable article URL used for the "open on Wikipedia" links."""

USER_AGENT: Final[str] = "WikiGraphExplorer/1.0 (educational project)"
"""``User-Agent`` header.

Wikipedia's robot policy asks every client to identify itself with a
descriptive ``User-Agent``; anonymous default agents are throttled [4]_.

.. [4] Wikimedia Foundation. (2024). *Wikimedia API etiquette*.
   https://foundation.wikimedia.org/wiki/Wikimedia_robots.txt
"""

REQUEST_TIMEOUT: Final[float] = 20.0
"""Per-request HTTP timeout in seconds."""

POLITE_DELAY: Final[float] = 0.1
"""Minimum delay between two consecutive API calls, in seconds."""

MAX_RETRIES: Final[int] = 4
"""Number of retries for transient failures (HTTP 429/5xx, timeouts)."""

RETRY_BACKOFF: Final[float] = 1.5
"""Base of the exponential backoff (seconds): ``RETRY_BACKOFF * 2**attempt``."""

LINK_NAMESPACE: Final[int] = 0
"""Only articles of the main namespace (0) are followed."""

LINK_PAGE_SIZE: Final[int] = 500
"""``pllimit`` value: the maximum number of links returned per API call."""

MAX_LINK_PAGES: Final[int] = 4
"""Maximum number of paginated ``prop=links`` calls per article.

The MediaWiki API returns links sorted by title, so reading the first few
pages yields a *deterministic* alphabetical prefix of the out-neighbourhood
without downloading the (potentially thousands of) remaining links.  Set to
``0`` to disable the cap and read every page.
"""

# --------------------------------------------------------------------------
# Crawl limits
# --------------------------------------------------------------------------
MIN_DEPTH: Final[int] = 1
"""Smallest accepted BFS depth."""

MAX_DEPTH: Final[int] = 3
"""Largest accepted BFS depth."""

DEFAULT_DEPTH: Final[int] = 2
"""Depth used when the user does not change the widget."""

DEFAULT_MAX_CHILDREN: Final[int] = 15
"""Default fan-out cap per article."""

MAX_MAX_CHILDREN: Final[int] = 60
"""Upper bound accepted for the fan-out widget."""

NODE_BUDGET_BY_DEPTH: Final[dict[int, int]] = {1: 60, 2: 200, 3: 1000}
"""Hard cap on the number of nodes kept in the graph, per BFS depth."""

# --------------------------------------------------------------------------
# Centrality hyper-parameters
# --------------------------------------------------------------------------
PAGERANK_ALPHA: Final[float] = 0.85
"""Damping factor ``alpha`` of the PageRank random walk [1]_.

``alpha = 0.85`` is the value originally proposed by Brin & Page and is the
default recommended by Page et al. for the (web-sized) graphs studied here.
"""

PAGERANK_MAX_ITER: Final[int] = 300
"""Maximum number of PageRank power iterations."""

PAGERANK_TOL: Final[float] = 1.0e-10
"""Convergence tolerance for the PageRank power iteration."""

HITS_MAX_ITER: Final[int] = 100
"""Maximum number of power iterations for HITS."""

BETWEENNESS_NORMALIZED: Final[bool] = True
"""Normalise betweenness by the number of node pairs (Brandes' ``O(N*M)``)."""

# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
COLOR_DEGREE_IN: Final[str] = "#e41a1c"
COLOR_DEGREE_OUT: Final[str] = "#ff7f00"
COLOR_BETWEENNESS: Final[str] = "#984ea3"
COLOR_CLOSENESS: Final[str] = "#4daf4a"
COLOR_EIGENVECTOR: Final[str] = "#a65628"
COLOR_PAGERANK: Final[str] = "#377eb8"
COLOR_HITS_AUTHORITY: Final[str] = "#e377c2"
COLOR_HITS_HUB: Final[str] = "#17becf"
COLOR_REGULAR: Final[str] = "#d9d9d9"
"""Colours of the per-measure "winner" node and of ordinary nodes."""

GRAPH_HEIGHT: Final[int] = 720
"""Height in pixels of the embedded pyvis iframe."""

GRAPH_WIDTH: Final[int] = 100
"""Width in percent of the embedded pyvis iframe."""

NODE_SIZE_MIN: Final[float] = 10.0
"""Smallest drawn node size (in px)."""

NODE_SIZE_MAX: Final[float] = 60.0
"""Largest drawn node size (in px)."""

# --------------------------------------------------------------------------
# Streamlit / server
# --------------------------------------------------------------------------
PAGE_TITLE: Final[str] = "Wiki Graph Centrality Explorer"
PAGE_ICON: Final[str] = "🕸️"
"""Favicon shown in the browser tab."""

PRIMARY_PORT: Final[int] = int(os.environ.get("STREAMLIT_PORT", "8080"))
"""Primary server port; overridable with the ``STREAMLIT_PORT`` env var."""

FALLBACK_PORT: Final[int] = int(os.environ.get("STREAMLIT_FALLBACK_PORT", "8501"))
"""Fallback server port used when the primary port is already bound."""

SERVER_ADDRESS: Final[str] = "0.0.0.0"
"""Bind address inside the container (0.0.0.0 so the port-forward works)."""

APP_TITLE: Final[str] = "Wikipedia Web-Graph Centrality Explorer"
APP_SUBTITLE: Final[str] = (
    "Social Network Analysis measures applied to a BFS subtree of Wikipedia"
)
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_LEVEL: Final[int] = logging.INFO


def setup_logging(level: int = LOG_LEVEL) -> logging.Logger:
    """Configure and return the package logger.

    A rotating-free, appending file handler writes every record to
    ``logs/app.log`` (INFO level) and a stream handler mirrors the messages to
    the container's stdout so that ``docker exec`` shows the crawl progress.

    Parameters
    ----------
    level
        Logging level for the root logger (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        The ``wge`` logger, ready to be used by the other modules.

    Examples
    --------
    >>> setup_logging().name
    'wge'
    """
    root = logging.getLogger()
    if not any(getattr(h, "_wge", False) for h in root.handlers):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        for handler in (file_handler, stream_handler):
            handler.setLevel(level)
            handler._wge = True  # type: ignore[attr-defined]
            root.addHandler(handler)
    root.setLevel(level)
    logger = logging.getLogger("wge")
    logger.setLevel(level)
    return logger
