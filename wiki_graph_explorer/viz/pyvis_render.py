"""pyvis rendering: node sizing, winner colouring and the node picker.

The module turns a :class:`networkx.DiGraph` plus a
:class:`graph.centrality.CentralityResult` into a self-contained HTML document
that Streamlit embeds with :func:`streamlit.components.v1.html`.  Node sizes
follow the measure chosen in the UI, and the top-ranked article of every
measure receives that measure's colour so that the eight rankings can be read
off the picture at a glance.

References
----------
.. [1] Langville, A. N., & Meyer, C. D. (2006). *Google's PageRank and Its
       Friends*. Now Publishers.
"""

from __future__ import annotations

import html
import json
import logging
from typing import Mapping, Sequence

import networkx as nx
from pyvis.network import Network

import config
from graph.centrality import (
    CLOSENESS,
    MEASURE_ORDER,
    MEASURES,
    PAGERANK,
    CentralityResult,
    normalise_sizes,
)

LOGGER = logging.getLogger("wge.viz.pyvis")

PICKER_ID = "wge-node-picker"
"""DOM id of the ``<select>`` element injected below the graph."""

ROOT_BORDER_COLOR = "#1f77b4"
"""Border colour used to mark the root article."""


def size_measure_options() -> list[str]:
    """Return the labels of the measures that can drive the node size.

    Returns
    -------
    list of str
        Display labels in canonical order (PageRank first, as the default).

    Examples
    --------
    >>> size_measure_options()[0]
    'PageRank'
    """
    ordered = [
        PAGERANK,
        CLOSENESS,
        "betweenness",
        "eigenvector",
        "hits_authority",
        "hits_hub",
        "degree_in",
        "degree_out",
    ]
    return [MEASURES[key].label for key in ordered if key in MEASURES]


def node_colors(centralities: CentralityResult) -> dict[str, str]:
    """Colour every node: winners get their measure colour, others light grey.

    A node that is top-ranked by several measures keeps the colour of the first
    measure in :data:`graph.centrality.MEASURE_ORDER`, so exactly one colour is
    drawn per node while every measure still highlights a winner.

    Parameters
    ----------
    centralities
        Computed centrality values.

    Returns
    -------
    dict
        ``{title: hex colour}`` for every article of the graph.

    Examples
    --------
    >>> colors = node_colors(centralities)   # doctest: +SKIP
    >>> colors["Information retrieval"]      # doctest: +SKIP
    '#377eb8'
    """
    winners = centralities.winners()
    # An article can win several measures; the first one in MEASURE_ORDER wins
    # the colour, later ones are dropped.
    colors: dict[str, str] = {}
    for key in MEASURE_ORDER:
        title = winners.get(key)
        if title is not None and title not in colors:
            colors[title] = MEASURES[key].color
    return colors


def build_network(
    graph: nx.DiGraph,
    centralities: CentralityResult,
    size_measure: str = PAGERANK,
) -> Network:
    """Build the pyvis :class:`~pyvis.network.Network` for a web graph.

    Parameters
    ----------
    graph
        Directed web graph.
    centralities
        All centrality values, used for sizing, colouring and tooltips.
    size_measure
        Key of the measure that drives the node size (default PageRank).

    Returns
    -------
    pyvis.network.Network
        Configured network object; call ``generate_html(notebook=False)`` on
        it to obtain the document.

    Examples
    --------
    >>> network = build_network(graph, centralities)   # doctest: +SKIP
    >>> len(network.nodes)
    148
    """
    sizes = normalise_sizes(centralities.values(size_measure))
    colors = node_colors(centralities)
    network = Network(
        height=f"{config.GRAPH_HEIGHT}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    network.set_options(
        json.dumps(
            {
                "layout": {"improvedLayout": True, "randomSeed": 42},
                "physics": {
                    "enabled": True,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -70,
                        "springLength": 130,
                    },
                    "stabilization": {"iterations": 220},
                },
                "interaction": {
                    "hover": True,
                    "navigationButtons": True,
                    "keyboard": False,
                },
                "edges": {
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                    "color": {"color": "#c8c8c8", "highlight": "#7f7f7f"},
                    "smooth": {"enabled": True, "type": "dynamic"},
                    "width": 1.0,
                },
                "nodes": {
                    "font": {"size": 12, "color": "#222222"},
                    "borderWidth": 1.5,
                    "shadow": False,
                },
            }
        )
    )
    for title in sorted(graph.nodes):
        attributes = graph.nodes[title]
        value = float(centralities.values(size_measure).get(title, 0.0))
        color = colors.get(title, config.COLOR_REGULAR)
        network.add_node(
            title,
            label=str(title),
            size=float(sizes.get(title, config.NODE_SIZE_MIN)),
            color=color,
            borderWidth=3.0 if attributes.get("is_root") else 1.5,
            borderWidthSelected=4.0,
            title=_tooltip(title, graph, centralities, size_measure, value),
        )
    for source, target in sorted(graph.edges):
        network.add_edge(source, target)
    LOGGER.info(
        "pyvis network built: %d nodes / %d edges (size by %s)",
        len(network.nodes),
        len(network.edges),
        size_measure,
    )
    return network


def render_html(
    graph: nx.DiGraph,
    centralities: CentralityResult,
    size_measure: str = PAGERANK,
    selected: str | None = None,
) -> str:
    """Render the interactive graph as a standalone HTML document.

    The generated document contains the inline pyvis/vis-network bundle, the
    network itself and an injected ``<select>`` picker (:data:`PICKER_ID`) that
    mirrors the graph selection.  Changing the picker - or clicking a node -
    writes the title into the ``node`` query parameter of the parent page,
    which is how the Streamlit details panel learns about the selection.

    Parameters
    ----------
    graph
        Directed web graph.
    centralities
        All centrality values.
    size_measure
        Key of the measure that drives the node size.
    selected
        Title to preselect in the picker, e.g. the article already shown in the
        details panel.

    Returns
    -------
    str
        A complete HTML document ready for
        ``st.components.v1.html(document, height=..., scrolling=True)``.

    Examples
    --------
    >>> document = render_html(graph, centralities, PAGERANK, "Networking")
    >>> "wge-node-picker" in document
    True
    """
    network = build_network(graph, centralities, size_measure)
    document = network.generate_html(notebook=False)
    return inject_node_picker(document, sorted(str(n) for n in graph.nodes), selected)


def inject_node_picker(
    document: str,
    titles: Sequence[str],
    selected: str | None = None,
) -> str:
    """Insert the article picker and its JavaScript into a pyvis document.

    Parameters
    ----------
    document
        HTML produced by ``Network.generate_html(notebook=False)``.
    titles
        Article titles to offer in the picker, already sorted.
    selected
        Title to preselect, if any.

    Returns
    -------
    str
        The same document with the picker markup, styles and script injected
        just before ``</body>``.

    Examples
    --------
    >>> "<select" in inject_node_picker("<body></body>", ["A"], "A")
    True
    """
    options = "".join(
        '<option value="{value}"{selected}>{label}</option>'.format(
            value=html.escape(title, quote=True),
            label=html.escape(title),
            selected=" selected" if title == selected else "",
        )
        for title in titles
    )
    placeholder = '<option value="">-- select an article --</option>'
    block = f"""
<style>
  #wge-picker-bar {{
    clear: both;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px;
    padding: 10px 4px 4px 4px;
    color: #31333f;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }}
  #wge-picker-bar label {{ font-weight: 600; }}
  #wge-picker-bar select {{
    min-width: 320px;
    max-width: 60%;
    padding: 6px 8px;
    border: 1px solid #d0d4dd;
    border-radius: 6px;
    background: #ffffff;
  }}
  #wge-picker-bar button {{
    padding: 6px 12px;
    border: 1px solid #d0d4dd;
    border-radius: 6px;
    background: #f6f7f9;
    cursor: pointer;
  }}
  #wge-picker-bar button:hover {{ background: #eceef2; }}
  #wge-hint {{ color: #6b7280; font-size: 12px; }}
</style>
<div id="wge-picker-bar">
  <label for="{PICKER_ID}">Inspect article:</label>
  <select id="{PICKER_ID}">{placeholder}{options}</select>
  <button id="wge-reset" type="button">Clear</button>
  <span id="wge-hint">Click a node in the graph, or pick an article above;
  the details panel follows the selection.</span>
</div>
<script>
(function () {{
  var picker = document.getElementById("{PICKER_ID}");
  if (!picker || typeof network === "undefined") {{ return; }}

  function focusNode(title) {{
    if (!title) {{ return; }}
    try {{
      if (network.selectNodes) {{ network.selectNodes([title]); }}
      else {{ network.selectNode([title]); }}
      network.focus(title, {{ scale: 1.15, animation: {{ duration: 350 }} }});
    }} catch (error) {{ /* node not part of the network */ }}
  }}

  function publish(title) {{
    try {{
      var url = new URL(window.parent.location.href);
      if (title) {{ url.searchParams.set("node", title); }}
      else {{ url.searchParams.delete("node"); }}
      window.parent.history.replaceState({{}}, "", url.toString());
      window.parent.dispatchEvent(new PopStateEvent("popstate"));
    }} catch (error) {{ /* cross-origin parent */ }}
  }}

  picker.addEventListener("change", function () {{
    focusNode(picker.value);
    publish(picker.value);
  }});

  var reset = document.getElementById("wge-reset");
  if (reset) {{
    reset.addEventListener("click", function () {{
      picker.selectedIndex = 0;
      publish("");
    }});
  }}

  network.on("click", function (params) {{
    if (params.nodes && params.nodes.length) {{
      picker.value = params.nodes[0];
      publish(params.nodes[0]);
    }}
  }});
  network.on("selectNode", function (params) {{
    if (params.nodes && params.nodes.length) {{ picker.value = params.nodes[0]; }}
  }});

  var initial = "";
  try {{ initial = new URL(window.parent.location.href).searchParams.get("node") || ""; }}
  catch (error) {{ initial = ""; }}
  if (initial) {{
    var known = Array.prototype.some.call(picker.options, function (option) {{
      return option.value === initial;
    }});
    if (known) {{ picker.value = initial; focusNode(initial); }}
  }}
}})();
</script>
"""
    if "</body>" not in document:
        return document + block
    return document.replace("</body>", block + "</body>", 1)


def _tooltip(
    title: str,
    graph: nx.DiGraph,
    centralities: CentralityResult,
    size_measure: str,
    value: float,
) -> str:
    """Build the hover tooltip of a node.

    Parameters
    ----------
    title
        Article title.
    graph
        Directed web graph (for the raw degrees).
    centralities
        All centrality values.
    size_measure
        Measure currently driving the node size.
    value
        The value of ``size_measure`` for this node.

    Returns
    -------
    str
        HTML snippet shown when hovering the node.

    Examples
    --------
    >>> _tooltip("A", graph, centralities, PAGERANK, 0.5).startswith("<b>")
    True
    """
    rows = "".join(
        f"<tr><td style='padding:0 8px 0 0'>{MEASURES[key].label}</td>"
        f"<td style='text-align:right'>{centralities.values(key).get(title, 0.0):.6f}</td></tr>"
        for key in MEASURE_ORDER
        if key in centralities.measures
    )
    return (
        f"<b>{html.escape(title)}</b><br/>"
        f"depth: {graph.nodes[title].get('depth', 0)} | "
        f"in: {graph.in_degree(title)} | out: {graph.out_degree(title)}<br/>"
        f"<i>size metric ({MEASURES[size_measure].label}): {value:.6f}</i>"
        f"<table>{rows}</table>"
    )


def winner_summary(centralities: CentralityResult) -> str:
    """Return a one-line summary of every measure winner.

    Parameters
    ----------
    centralities
        All centrality values.

    Returns
    -------
    str
        Semicolon separated ``"measure: article"`` pairs.

    Examples
    --------
    >>> "PageRank" in winner_summary(centralities)   # doctest: +SKIP
    True
    """
    return "; ".join(
        f"{MEASURES[key].label}: {title}"
        for key, title in centralities.winners().items()
    )


def centrality_json(centralities: CentralityResult) -> str:
    """Serialise the centrality values for the frontend.

    Parameters
    ----------
    centralities
        All centrality values.

    Returns
    -------
    str
        JSON object ``{measure key: {title: value}}``.

    Examples
    --------
    >>> json.loads(centrality_json(centralities))["pagerank"]  # doctest: +SKIP
    {'Information retrieval': 0.01, ...}
    """
    return json.dumps(
        {key: values for key, values in centralities.measures.items()},
        ensure_ascii=False,
        sort_keys=True,
    )


def measure_values(
    centralities: CentralityResult, measure_key: str
) -> Mapping[str, float]:
    """Return the values of one measure.

    Parameters
    ----------
    centralities
        All centrality values.
    measure_key
        One of :data:`graph.centrality.MEASURE_ORDER`.

    Returns
    -------
    collections.abc.Mapping
        ``{title: value}``; empty when the key is unknown.
    """
    return centralities.values(measure_key)
