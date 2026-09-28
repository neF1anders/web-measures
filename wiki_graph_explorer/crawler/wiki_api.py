"""MediaWiki API client used to crawl outgoing article links.

The client is deliberately small and dependency-light: it wraps a
``requests.Session`` with a descriptive ``User-Agent``, a politeness delay, an
exponential-backoff retry policy for HTTP 429/5xx, and a content-addressed
JSON cache on disk so that repeated runs of the demo do not hit Wikipedia
again.

References
----------
.. [1] MediaWiki Action API. (2024). *prop=links*.
       https://www.mediawiki.org/wiki/API:Links
.. [2] Wikimedia Foundation. (2024). *API etiquette*.
       https://foundation.wikimedia.org/wiki/Wikimedia_robots.txt
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

import config

LOGGER = logging.getLogger("wge.crawler.api")

DISAMBIGUATION_SUFFIX = "(disambiguation)"
"""Titles ending with this suffix are skipped: they are index pages, not
concepts, and they would otherwise dominate the betweenness ranking."""

TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
"""HTTP status codes that are worth retrying."""


class WikiApiError(RuntimeError):
    """Raised when the MediaWiki API cannot be queried successfully."""


@dataclass(frozen=True)
class ArticleRef:
    """A resolved (language, title) pair pointing at one Wikipedia article.

    Attributes
    ----------
    lang
        Wikipedia language edition code, e.g. ``"en"`` or ``"ru"``.
    title
        Canonical article title with underscores replaced by spaces.
    """

    lang: str
    title: str

    @property
    def url(self) -> str:
        """Canonical ``https://{lang}.wikipedia.org/wiki/{title}`` URL.

        Returns
        -------
        str
            The percent-encoded article URL.
        """
        return article_url(self.lang, self.title)


def normalise_title(title: str) -> str:
    """Normalise a Wikipedia title (underscores to spaces, collapsed spaces).

    Parameters
    ----------
    title
        Raw title as returned by the API or typed by the user.

    Returns
    -------
    str
        The normalised title.

    Examples
    --------
    >>> normalise_title("Information_retrieval")
    'Information retrieval'
    """
    return " ".join(urllib.parse.unquote(title).replace("_", " ").split())


def article_url(lang: str, title: str) -> str:
    """Build the canonical browser URL of an article.

    Parameters
    ----------
    lang
        Wikipedia language edition code.
    title
        Article title.

    Returns
    -------
    str
        Fully qualified, percent-encoded article URL.

    Examples
    --------
    >>> article_url("en", "Information retrieval")
    'https://en.wikipedia.org/wiki/Information%20retrieval'
    """
    quoted = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return config.ARTICLE_URL_TEMPLATE.format(lang=lang, title=quoted)


def cache_key(lang: str, title: str) -> str:
    """Return the on-disk cache file name for an article.

    Parameters
    ----------
    lang
        Wikipedia language edition code.
    title
        Article title.

    Returns
    -------
    str
        File name of the form ``{lang}_{sha1(title)}.json``.
    """
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()
    return f"{lang}_{digest}.json"


def looks_like_disambiguation(title: str) -> bool:
    """Heuristically detect pages that are disambiguation indexes.

    Parameters
    ----------
    title
        Candidate article title.

    Returns
    -------
    bool
        ``True`` when the title is a disambiguation page or a fragment link.

    Examples
    --------
    >>> looks_like_disambiguation("Mercury (disambiguation)")
    True
    """
    lowered = title.casefold()
    return lowered.endswith(DISAMBIGUATION_SUFFIX) or title.startswith("#")


def parse_article_input(text: str, default_lang: str = "en") -> ArticleRef:
    """Parse user input into an :class:`ArticleRef`.

    Accepted formats (in order of detection):

    1. Full article URL - ``https://ru.wikipedia.org/wiki/Информационный_поиск``
    2. Mobile / index URLs - ``https://en.m.wikipedia.org/wiki/Foo`` and
       ``https://en.wikipedia.org/w/index.php?title=Foo``
    3. Language-prefixed title - ``ru:Информационный поиск`` or ``ru Информационный поиск``
    4. Bare title - ``Information retrieval`` (uses ``default_lang``)

    Parameters
    ----------
    text
        Raw text typed in the "Wikipedia article URL" input field.
    default_lang
        Language edition used when the input carries no language information.

    Returns
    -------
    ArticleRef
        The parsed language and normalised title.

    Raises
    ------
    ValueError
        If ``text`` is empty or cannot be interpreted as a Wikipedia article.

    Examples
    --------
    >>> parse_article_input("https://en.wikipedia.org/wiki/Information_retrieval")
    ArticleRef(lang='en', title='Information retrieval')
    >>> parse_article_input("ru:Информационный_поиск")
    ArticleRef(lang='ru', title='Информационный поиск')
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Please enter a Wikipedia article URL or title.")

    lowered = raw.lower()
    if "wikipedia.org/wiki/" in lowered or "wikipedia.org/w/" in lowered:
        parts = urllib.parse.urlparse(raw if "://" in raw else f"https://{raw}")
        lang = parts.netloc.split(".")[0] or default_lang
        if "index.php" in parts.path:
            title = urllib.parse.parse_qs(parts.query).get("title", [""])[0]
        else:
            title = parts.path.split("/wiki/", 1)[-1].split("/w/", 1)[0]
        title = normalise_title(title)
        if not title:
            raise ValueError("Could not read an article title from that URL.")
        return ArticleRef(lang=lang, title=title)

    for separator in (":", " ", "|"):
        prefix, found, rest = raw.partition(separator)
        if found and len(prefix) in (2, 3) and prefix.isalpha() and rest.strip():
            return ArticleRef(lang=prefix.lower(), title=normalise_title(rest))
    return ArticleRef(lang=default_lang, title=normalise_title(raw))


class WikiApiClient:
    """Small, polite, caching client for the MediaWiki ``api.php`` endpoint.

    Parameters
    ----------
    lang
        Wikipedia language edition code used to build the endpoint URL.
    session
        Optional pre-built :class:`requests.Session` (injected by the tests).
    cache_dir
        Directory for the JSON response cache; ``None`` disables caching.
    delay
        Minimum delay in seconds between two API calls.
    timeout
        Per-request timeout in seconds.
    max_retries
        Number of retries for transient HTTP failures.
    max_link_pages
        Maximum number of paginated ``prop=links`` calls per article.

    Examples
    --------
    >>> client = WikiApiClient("en", cache_dir=None, delay=0.0)  # doctest: +SKIP
    >>> client.fetch_links("Information retrieval", 3)           # doctest: +SKIP
    ['1890 US census', '3D retrieval', 'Academic freedom']
    """

    def __init__(
        self,
        lang: str = "en",
        session: requests.Session | None = None,
        cache_dir: Path | None = config.CACHE_DIR,
        delay: float = config.POLITE_DELAY,
        timeout: float = config.REQUEST_TIMEOUT,
        max_retries: int = config.MAX_RETRIES,
        max_link_pages: int = config.MAX_LINK_PAGES,
    ) -> None:
        self.lang = lang
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.delay = float(delay)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.max_link_pages = int(max_link_pages)
        self.stats: dict[str, int] = {"fetched": 0, "cached": 0, "failed": 0}
        self._last_request_at = 0.0
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": config.USER_AGENT,
                "Accept": "application/json",
            }
        )

    # -- public API --------------------------------------------------------
    @property
    def api_url(self) -> str:
        """The ``api.php`` URL this client talks to.

        Returns
        -------
        str
            Formatted API endpoint for :attr:`lang`.
        """
        return config.API_URL_TEMPLATE.format(lang=self.lang)

    def article_url(self, title: str) -> str:
        """Return the browser URL of ``title`` on this language edition.

        Parameters
        ----------
        title
            Article title.

        Returns
        -------
        str
            Percent-encoded article URL.
        """
        return article_url(self.lang, title)

    def resolve_title(self, title: str) -> str:
        """Resolve redirects and normalisation for a single title.

        Parameters
        ----------
        title
            Possibly unnormalised or redirecting title.

        Returns
        -------
        str
            The canonical title as known to Wikipedia; the normalised input is
            returned unchanged when the API is unreachable.
        """
        payload = self._request(
            {
                "action": "query",
                "format": "json",
                "titles": title,
                "redirects": 1,
                "prop": "info",
                "inprop": "url",
            }
        )
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                LOGGER.warning("Article not found: %s", title)
                return normalise_title(title)
            return normalise_title(page.get("title", title))
        return normalise_title(title)

    def fetch_links(
        self, title: str, max_children: int = config.DEFAULT_MAX_CHILDREN
    ) -> list[str]:
        """Return up to ``max_children`` outgoing links of ``title``.

        Only links into the main namespace (0) are returned, disambiguation
        pages are dropped, and the result is sorted by title so that the graph
        is byte-for-byte reproducible across runs [1]_.

        Parameters
        ----------
        title
            Article to inspect.
        max_children
            Maximum number of out-edges to keep.

        Returns
        -------
        list of str
            Deterministically sorted titles of the out-neighbours.

        Raises
        ------
        WikiApiError
            If the article does not exist or the API keeps failing.
        """
        cache_file = self._cache_path(title)
        cached = self._read_cache(cache_file)
        if cached is not None:
            self.stats["cached"] += 1
            LOGGER.debug("Cache hit for %s", title)
            return self._truncate(cached, max_children)

        collected, exhausted = self._collect_links(title, max_children)
        if not collected:
            self._write_cache(cache_file, [])
            raise WikiApiError(f"No main-namespace links found for '{title}'.")
        self._write_cache(cache_file, collected)
        self.stats["fetched"] += 1
        LOGGER.info("Fetched %d links for %s", len(collected), title)
        if not exhausted:
            LOGGER.info(
                "Link list for '%s' truncated by MAX_LINK_PAGES=%d",
                title,
                self.max_link_pages,
            )
        return self._truncate(collected, max_children)

    # -- internals ---------------------------------------------------------
    def _truncate(self, titles: Sequence[str], max_children: int) -> list[str]:
        """Sort and clip a list of titles to ``max_children`` entries.

        Parameters
        ----------
        titles
            Raw titles collected from the API.
        max_children
            Maximum number of titles to keep.

        Returns
        -------
        list of str
            Sorted, de-duplicated, clipped titles.
        """
        return sorted(set(titles))[: max(0, int(max_children))]

    def _collect_links(self, title: str, min_links: int) -> tuple[list[str], bool]:
        """Page through ``prop=links`` until ``min_links`` titles are gathered.

        Parameters
        ----------
        title
            Article to inspect.
        min_links
            Stop as soon as at least this many titles were collected, which
            makes the result equal to the alphabetical prefix of the article.

        Returns
        -------
        tuple
            ``(titles, exhausted)`` where ``exhausted`` is ``True`` when the
            whole link list was read.

        Raises
        ------
        WikiApiError
            When the article is missing or the request fails permanently.
        """
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "prop": "links",
            "titles": title,
            "plnamespace": config.LINK_NAMESPACE,
            "pllimit": config.LINK_PAGE_SIZE,
        }
        titles: list[str] = []
        missing = False
        for page_number in range(1, max(1, self.max_link_pages) + 1):
            payload = self._request(params)
            for page in payload.get("query", {}).get("pages", {}).values():
                if "missing" in page:
                    missing = True
                    continue
                for link in page.get("links", []) or []:
                    link_title = normalise_title(str(link.get("title", "")))
                    if link_title and not looks_like_disambiguation(link_title):
                        titles.append(link_title)
            continuation = payload.get("continue")
            if missing or not continuation or len(titles) >= min_links:
                return sorted(set(titles)), not bool(continuation)
            params = {**params, **{k: str(v) for k, v in continuation.items()}}
        return sorted(set(titles)), False

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perform one API call with rate limiting and retries.

        Parameters
        ----------
        params
            Query string parameters for ``api.php``.

        Returns
        -------
        dict
            Decoded JSON response.

        Raises
        ------
        WikiApiError
            If every attempt failed.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._sleep_if_needed()
            try:
                response = self.session.get(
                    self.api_url, params=params, timeout=self.timeout
                )
                self._last_request_at = time.monotonic()
            except requests.RequestException as error:  # pragma: no cover
                last_error = error
                LOGGER.warning(
                    "Request error (%s/%s): %s", attempt + 1, self.max_retries, error
                )
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as error:
                        last_error = error
                elif response.status_code in TRANSIENT_STATUS:
                    last_error = WikiApiError(f"HTTP {response.status_code}")
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(min(30.0, float(retry_after)))
                else:
                    self.stats["failed"] += 1
                    raise WikiApiError(
                        f"HTTP {response.status_code} for {params.get('titles')}"
                    )
            if attempt < self.max_retries:
                time.sleep(config.RETRY_BACKOFF * (2**attempt))
        self.stats["failed"] += 1
        raise WikiApiError(f"MediaWiki request failed: {last_error}") from last_error

    def _sleep_if_needed(self) -> None:
        """Sleep so that at least :attr:`delay` separates two API calls."""
        if self.delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _cache_path(self, title: str) -> Path | None:
        """Return the cache file path for ``title`` (``None`` if disabled).

        Parameters
        ----------
        title
            Article title.

        Returns
        -------
        pathlib.Path or None
            Absolute path of the cache entry.
        """
        if self.cache_dir is None:
            return None
        return self.cache_dir / cache_key(self.lang, title)

    def _read_cache(self, path: Path | None) -> list[str] | None:
        """Read a cached link list.

        Parameters
        ----------
        path
            Cache file path.

        Returns
        -------
        list of str or None
            Cached titles, or ``None`` on a miss / corrupted entry.
        """
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOGGER.warning("Ignoring corrupted cache entry %s", path)
            return None
        links: Iterable[str] = payload.get("links", [])
        return [str(item) for item in links]

    def _write_cache(self, path: Path | None, titles: Sequence[str]) -> None:
        """Store a link list on disk, ignoring write errors.

        The cache is a courtesy mechanism: if the cache directory cannot be
        written the crawler simply re-fetches the article next time.

        Parameters
        ----------
        path
            Cache file path, or ``None`` when caching is disabled.
        titles
            Titles to store (already sorted and de-duplicated).
        """
        if path is None:
            return
        payload = {
            "lang": self.lang,
            "links": list(titles),
            "fetched_at": int(time.time()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as error:  # pragma: no cover - depends on the filesystem
            LOGGER.warning("Could not write cache entry %s: %s", path, error)
