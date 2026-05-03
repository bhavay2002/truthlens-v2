from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

# G-V1 / G-D3: the canonical graph type produced by every other
# module in ``src.graph.*`` is ``Dict[str, Dict[str, float]]`` (a
# weighted adjacency dict). The visualizer was typed as
# ``Dict[str, List[str]]`` and ``_validate_graph`` rejected the
# canonical form outright — calling ``GraphVisualizer.visualize`` on
# any pipeline output raised ``ValueError("Invalid graph format")``,
# making the visualizer unreachable from the live pipeline. We now
# accept either shape and normalise in ``_to_nx``.
GraphInput = Union[
    Mapping[str, List[str]],
    Mapping[str, Mapping[str, float]],
]

def ensure_headless_matplotlib_backend() -> None:
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

ensure_headless_matplotlib_backend()

import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)
EPS = 1e-12

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


class GraphVisualizer:

    def __init__(self, output_dir: str | Path = "reports/graphs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("GraphVisualizer initialized")

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_graph(self, graph: GraphInput) -> None:
        # G-V1: accept both the legacy ``Dict[str, List[str]]`` and the
        # canonical weighted ``Dict[str, Dict[str, float]]`` produced by
        # the rest of the pipeline. Previously rejected the latter
        # outright, which made the visualizer unreachable from
        # production.
        if not isinstance(graph, dict):
            raise ValueError("graph must be dict")
        for k, v in graph.items():
            if not isinstance(k, str):
                raise ValueError("Invalid graph format: keys must be str")
            if not isinstance(v, (list, dict)):
                raise ValueError(
                    "Invalid graph format: neighbours must be list or dict"
                )

    # =====================================================
    # GRAPH BUILD
    # =====================================================

    def _to_nx(self, graph: GraphInput) -> nx.Graph:
        G = nx.Graph()
        for node, nbrs in graph.items():
            n = node.strip().lower()
            G.add_node(n)

            # Canonical weighted form: dict-of-dict.
            if isinstance(nbrs, dict):
                for nbr, w in nbrs.items():
                    if isinstance(nbr, str):
                        m = nbr.strip().lower()
                        if m and m != n:
                            G.add_edge(n, m, weight=float(w))
            # Legacy unweighted form: list of neighbour ids.
            else:
                for nbr in nbrs:
                    if isinstance(nbr, str):
                        m = nbr.strip().lower()
                        if m and m != n:
                            G.add_edge(n, m, weight=1.0)
        return G

    # =====================================================
    # NODE IMPORTANCE
    # =====================================================

    def _node_importance(
        self,
        G: nx.Graph,
        importance: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:

        if importance:
            values = np.array(
                [importance.get(n, 0.0) for n in G.nodes()],
                dtype=float,
            )
        else:
            centrality = nx.degree_centrality(G)
            values = np.array(list(centrality.values()), dtype=float)

        if values.size == 0:
            return values

        return values / (np.max(values) + EPS)

    # =====================================================
    # EDGE WEIGHTS
    # =====================================================

    def _edge_widths(
        self,
        G: nx.Graph,
        edge_importance: Optional[Dict[str, float]] = None,
    ):

        widths = []

        for u, v in G.edges():
            key = f"{u}->{v}"
            rev = f"{v}->{u}"

            val = 1.0

            if edge_importance:
                val = max(
                    edge_importance.get(key, 0.0),
                    edge_importance.get(rev, 0.0),
                    0.1,
                )

            widths.append(1.0 + 5.0 * val)

        return widths

    # =====================================================
    # TEMPORAL COLORING
    # =====================================================

    def _temporal_colors(
        self,
        G: nx.Graph,
        temporal_features: Optional[Dict[str, float]],
    ):

        if not temporal_features:
            return None

        drift = float(temporal_features.get("narrative_drift", 0.0))

        return np.full(G.number_of_nodes(), drift)

    # =====================================================
    # STATIC DRAW
    # =====================================================

    def _draw_static(
        self,
        G: nx.Graph,
        title: str,
        path: Path,
        *,
        node_importance: Optional[Dict[str, float]] = None,
        edge_importance: Optional[Dict[str, float]] = None,
        temporal_features: Optional[Dict[str, float]] = None,
    ) -> Path:

        plt.figure(figsize=(12, 10))

        pos = nx.spring_layout(G, seed=42, k=0.5)

        node_colors = self._node_importance(G, node_importance)

        # temporal override (if present)
        temporal_colors = self._temporal_colors(G, temporal_features)
        if temporal_colors is not None:
            node_colors = temporal_colors

        edge_widths = self._edge_widths(G, edge_importance)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=500,
            node_color=node_colors if len(node_colors) else "blue",
            cmap="Reds",
        )

        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            alpha=0.7,
        )

        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(title)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

        return path

    # =====================================================
    # INTERACTIVE
    # =====================================================

    def _draw_interactive(
        self,
        G: nx.Graph,
        node_importance: Optional[Dict[str, float]] = None,
    ):

        if go is None:
            raise ImportError("Plotly not installed")

        pos = nx.spring_layout(G, seed=42)

        edge_x, edge_y = [], []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1),
            hoverinfo="none",
        )

        node_x, node_y, text, color = [], [], [], []

        importance_vec = self._node_importance(G, node_importance)

        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(node)
            color.append(importance_vec[i] if len(importance_vec) else 0.5)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=text,
            textposition="top center",
            marker=dict(
                size=12,
                color=color,
                colorscale="Reds",
            ),
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.show()

    # =====================================================
    # PUBLIC API
    # =====================================================

    def visualize(
        self,
        graph: GraphInput,
        *,
        name: str = "graph",
        interactive: bool = False,
        node_importance: Optional[Dict[str, float]] = None,
        edge_importance: Optional[Dict[str, float]] = None,
        temporal_features: Optional[Dict[str, float]] = None,
    ) -> Path:

        self._validate_graph(graph)

        G = self._to_nx(graph)

        if interactive:
            self._draw_interactive(G, node_importance)

        path = self.output_dir / f"{name}.png"

        return self._draw_static(
            G,
            f"{name.title()} Graph",
            path,
            node_importance=node_importance,
            edge_importance=edge_importance,
            temporal_features=temporal_features,
        )