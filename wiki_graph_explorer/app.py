"""Streamlit entry point of the Wiki Graph Centrality Explorer.

The module is a thin orchestrator: it wires the sidebar widgets to the crawler,
computes the centrality measures, renders the pyvis graph and shows the details
panel.  All logic lives in :mod:`crawler`, :mod:`graph`, :mod:`viz` and
:mod:`ui`.

Run it inside the container with either of::

    docker exec -it ml-workspace python app.py
    docker exec -it ml-workspace python /workspace/copybook/WebSearch/wiki_graph_explorer/app.py

``python app.py`` re-executes the file with ``streamlit run`` on port
``$STREAMLIT_PORT`` (default 8080) and falls back to 8501 when 8080 is busy.

References
----------
.. [1] Newman, M. E. J. (2005). *Networks: An Introduction*. Oxford
       University Press.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import requests
import streamlit as st
import streamlit.components.v1 as components

import config
from crawler.bfs import CrawlProgress, CrawlResult, build_crawl
from crawler.wiki_api import WikiApiClient, WikiApiError
from graph.builder import build_digraph, to_table_rows
from graph.centrality import (
    MEASURES,
    PAGERANK,
    CentralityResult,
    compute_all_centralities,
    measure_keys,
)
from ui.details_panel import render_details_panel
from ui.sidebar import CrawlRequest, render_controls, render_legend_panel
from viz.pyvis_render import render_html, size_measure_options

LOGGER = logging.getLogger("wge.app")

STATE_KEY = "wge_graph_state"
NODE_KEY = "wge_node"
PENDING_NODE_KEY = "wge_pending_node"
SIZE_KEY = "wge_size_measure"
PARAM_KEY = "node"

HIDE_CHROME_CSS = """
<style>
[data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"] {display: none; visibility: hidden;}
.block-container {padding-top: 2.2rem;}
</style>
"""


@dataclass
class GraphState:
    """Everything computed for one crawl, cached in ``st.session_state``.

    Attributes
    ----------
    crawl
        Output of the BFS crawler.
    graph
        The directed web graph.
    centralities
        All eight centrality vectors.
    """

    crawl: CrawlResult
    graph: nx.DiGraph
    centralities: CentralityResult


# ---------------------------------------------------------------- orchestration
def _crawl_state(request: CrawlRequest) -> GraphState:
    """Crawl, build the graph and compute the centralities for one request.

    Parameters
    ----------
    request
        Validated :class:`ui.sidebar.CrawlRequest`.

    Returns
    -------
    GraphState
        The freshly computed state.

    Notes
    -----
    ``st.progress`` and ``st.empty`` write to Streamlit placeholders, so the
    crawl counters are visible in the browser while the crawl is running.
    """
    client = WikiApiClient(lang=request.ref.lang)
    try:
        resolved = client.resolve_title(request.ref.title)
    except WikiApiError as error:
        st.warning(f"Title could not be resolved ({error}); using it verbatim.")
        resolved = request.ref.title
    status = st.empty()
    progress_bar = st.progress(0.0, text="Crawling Wikipedia ...")

    def on_progress(snapshot: CrawlProgress) -> None:
        fraction = min(1.0, snapshot.fetched / max(1, snapshot.total))
        progress_bar.progress(fraction, text=snapshot.as_text())
        status.info(f"**{snapshot.as_text()}**")

    crawl = build_crawl(
        client,
        resolved,
        depth=request.depth,
        max_children=request.max_children,
        progress=on_progress,
    )
    progress_bar.progress(1.0, text=f"Done: {crawl.node_count} articles")
    status.empty()
    graph = build_digraph(crawl)
    with st.spinner("Computing centrality measures ..."):
        centralities = compute_all_centralities(graph)
    return GraphState(crawl=crawl, graph=graph, centralities=centralities)


def _sync_selection(state: GraphState) -> str:
    """Resolve the selected article from the pending value, URL and defaults.

    The pyvis iframe writes the picked node into the ``node`` query parameter,
    so reading it back here is what turns a click inside the graph into a
    Streamlit rerun with an updated details panel.  Selections that arrive from
    the dataframe below the graph are staged in :data:`PENDING_NODE_KEY`
    because a widget value may not be modified after its widget was created.

    Parameters
    ----------
    state
        The current :class:`GraphState`.

    Returns
    -------
    str
        The selected article title.
    """
    nodes = state.graph.nodes
    pending = st.session_state.pop(PENDING_NODE_KEY, "")
    from_url = st.query_params.get(PARAM_KEY, "")
    for candidate in (pending, from_url):
        if (
            candidate
            and candidate in nodes
            and candidate != st.session_state.get(NODE_KEY)
        ):
            st.session_state[NODE_KEY] = candidate
            break
    if NODE_KEY not in st.session_state or st.session_state[NODE_KEY] not in nodes:
        st.session_state[NODE_KEY] = state.crawl.root
    return str(st.session_state[NODE_KEY])


def _on_node_selected() -> None:
    """Push the dropdown selection into the URL read by the pyvis iframe."""
    st.query_params[PARAM_KEY] = str(st.session_state.get(NODE_KEY, ""))


def _render_graph_area(state: GraphState) -> None:
    """Render the graph iframe, the node picker and the details panel.

    Parameters
    ----------
    state
        The current :class:`GraphState`.
    """
    selected = _sync_selection(state)
    head, picker = st.columns(2)
    with head:
        labels = size_measure_options()
        st.selectbox(
            "Node size metric",
            labels,
            index=labels.index(MEASURES[PAGERANK].label),
            key=SIZE_KEY,
        )
    with picker:
        st.selectbox(
            "Inspect article",
            sorted(state.graph.nodes, key=str),
            key=NODE_KEY,
            on_change=_on_node_selected,
        )
    size_measure = measure_keys([st.session_state[SIZE_KEY]])[0]
    graph_column, panel_column = st.columns([2, 1], gap="large")
    with graph_column:
        components.html(
            render_html(state.graph, state.centralities, size_measure, selected),
            height=config.GRAPH_HEIGHT + 140,
            scrolling=True,
        )
    with panel_column:
        render_details_panel(state.graph, state.centralities, selected)


def _render_node_table(state: GraphState) -> None:
    """Render the searchable table of all articles.

    Parameters
    ----------
    state
        The current :class:`GraphState`.

    Notes
    -----
    A row click stages the title in :data:`PENDING_NODE_KEY`; the next rerun
    applies it to the details panel and to the dropdown above the graph.
    """
    size_measure = measure_keys([st.session_state[SIZE_KEY]])[0]
    rows = to_table_rows(state.graph, state.centralities, size_measure)
    titles = [row["Article"] for row in rows]
    with st.expander(
        f"All {len(titles)} articles (searchable, click a row)", expanded=False
    ):
        st.caption(
            f"Sorted by {MEASURES[size_measure].label}; click a row to load it "
            "into the details panel."
        )
        event = st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="wge_table",
            column_config={
                MEASURES[key].label: st.column_config.NumberColumn(format="%.6f")
                for key in state.centralities.measures
            },
        )
        chosen = [int(index) for index in getattr(event.selection, "rows", [])]
        if chosen and 0 <= chosen[0] < len(titles):
            title = titles[chosen[0]]
            st.session_state[PENDING_NODE_KEY] = title
            st.query_params[PARAM_KEY] = title


def _render_header(state: GraphState) -> None:
    """Render the title, the KPI row and any crawl warnings.

    Parameters
    ----------
    state
        The current :class:`GraphState`.
    """
    st.title(config.APP_TITLE)
    st.caption(
        f"{config.APP_SUBTITLE} &middot; root: **{state.crawl.root}** "
        f"(`{state.crawl.lang}.wikipedia.org`)"
    )
    metrics = st.columns(5)
    metrics[0].metric("Articles", state.crawl.node_count)
    metrics[1].metric("Links", state.crawl.edge_count)
    metrics[2].metric("Depth", state.crawl.depth)
    metrics[3].metric("Max children", state.crawl.max_children)
    metrics[4].metric("Language", state.crawl.lang)
    if state.crawl.truncated:
        st.warning(
            "The node budget was reached: the graph was truncated, so the "
            "centrality values describe the crawled sub-graph only.",
            icon="⚠️",
        )
    if state.crawl.errors:
        failed = ", ".join(sorted(state.crawl.errors)[:5])
        st.warning(f"{len(state.crawl.errors)} article(s) could not be read: {failed}")
    if not state.graph.number_of_edges():
        st.error("No links were found - nothing to analyse.")


def render_app() -> None:
    """Run the whole Streamlit script (one page render / rerun)."""
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "about": config.APP_SUBTITLE,
            "Get Help": None,
            "Report a bug": None,
        },
    )
    st.markdown(HIDE_CHROME_CSS, unsafe_allow_html=True)
    request = render_controls()
    if request is not None:
        try:
            state = _crawl_state(request)
            st.session_state[STATE_KEY] = state
            st.query_params[PARAM_KEY] = state.crawl.root
        except (WikiApiError, requests.RequestException) as error:
            st.session_state.pop(STATE_KEY, None)
            st.error(f"Wikipedia could not be crawled: {error}")
    state: GraphState | None = st.session_state.get(STATE_KEY)
    if state is None:
        st.info("Enter a Wikipedia article in the sidebar and press **Build graph**.")
    else:
        _render_header(state)
        _render_graph_area(state)
        _render_node_table(state)
    render_legend_panel(state.centralities if state else None)


# -------------------------------------------------------------------- launcher
def _is_streamlit_runtime() -> bool:
    """Return ``True`` when this file is executed by ``streamlit run``.

    Returns
    -------
    bool
        ``True`` inside a Streamlit script run, ``False`` under plain Python.

    Examples
    --------
    >>> _is_streamlit_runtime() in (True, False)
    True
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # pragma: no cover - depends on Streamlit internals
        return False


def _port_is_free(port: int) -> bool:
    """Check whether ``port`` can still be bound on ``0.0.0.0``.

    Parameters
    ----------
    port
        TCP port number.

    Returns
    -------
    bool
        ``True`` when the port is available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def resolve_port(preferred: int, fallback: int) -> int:
    """Return the first free port out of ``preferred`` and ``fallback``.

    Parameters
    ----------
    preferred
        Primary port, normally 8080.
    fallback
        Secondary port, normally 8501.

    Returns
    -------
    int
        The selected port; ``preferred`` when both ports are busy.
    """
    for port in (preferred, fallback):
        if _port_is_free(port):
            if port != preferred:
                LOGGER.warning("Port %d busy, falling back to %d", preferred, port)
            return port
    return preferred


def launch_streamlit(port: int) -> int:
    """Start ``streamlit run app.py`` in a subprocess and wait for it.

    Parameters
    ----------
    port
        Port the Streamlit server should listen on.

    Returns
    -------
    int
        Exit code of the Streamlit process.
    """
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(port),
        "--server.address",
        config.SERVER_ADDRESS,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("=" * 72)
    print(f"  {config.APP_TITLE}")
    print(f"  serving on http://localhost:{port}  (ctrl-c to stop)")
    print("=" * 72)
    environment = {**os.environ, "WGE_LAUNCHED": "1"}
    return subprocess.run(
        command, env=environment, cwd=str(config.BASE_DIR), check=False
    ).returncode


def main() -> None:
    """Entry point: run the UI, or start the Streamlit server if needed."""
    config.setup_logging()
    if _is_streamlit_runtime() or os.environ.get("WGE_LAUNCHED") == "1":
        render_app()
        return
    port = resolve_port(config.PRIMARY_PORT, config.FALLBACK_PORT)
    raise SystemExit(launch_streamlit(port))


if __name__ == "__main__":
    main()
