"""Interactive node-link graph rendering for the web graph."""

from __future__ import annotations

from .legends import legend_html, measure_reference_markdown
from .pyvis_render import (
    build_network,
    inject_node_picker,
    node_colors,
    render_html,
    size_measure_options,
    winner_summary,
)

__all__ = [
    "build_network",
    "inject_node_picker",
    "legend_html",
    "measure_reference_markdown",
    "node_colors",
    "render_html",
    "size_measure_options",
    "winner_summary",
]
