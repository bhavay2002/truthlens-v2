from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


class ExplanationVisualizer:

    def __init__(self) -> None:
        logger.info("ExplanationVisualizer initialized")

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def _validate(tokens: List[str], scores: List[float]) -> None:
        if not tokens or not scores:
            raise ValueError("tokens and scores must not be empty")
        if len(tokens) != len(scores):
            raise ValueError("tokens and scores must match length")

    # =====================================================
    # NORMALIZATION
    # =====================================================

    @staticmethod
    def _normalize(scores: List[float]) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        s = np.abs(s)
        return s / (np.sum(s) + EPS)

    # =====================================================
    # SAVE HANDLER
    # =====================================================

    def _finalize(self, fig, save_path: Optional[str]):
        import matplotlib.pyplot as plt  # GPU-5: lazy import
        if save_path:
            fig.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    # =====================================================
    #  MULTI-METHOD OVERLAY (NEW)
    # =====================================================

    def plot_multi_method_overlay(
        self,
        tokens: List[str],
        explanations: Dict[str, List[float]],
        *,
        normalize: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Overlay multiple explanation methods on same token axis.
        """

        for scores in explanations.values():
            self._validate(tokens, scores)

        if normalize:
            explanations = {
                k: self._normalize(v) for k, v in explanations.items()
            }

        import matplotlib.pyplot as plt  # GPU-5: lazy import
        fig, ax = plt.subplots(figsize=(12, 6))

        for name, scores in explanations.items():
            ax.plot(tokens, scores, marker="o", label=name)

        ax.set_title("Multi-Method Explanation Overlay")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        self._finalize(fig, save_path)

    # =====================================================
    #  PLOTLY INTERACTIVE (NEW)
    # =====================================================

    def plot_interactive(
        self,
        tokens: List[str],
        explanations: Dict[str, List[float]],
        *,
        normalize: bool = True,
    ):
        """
        Interactive Plotly visualization.
        """

        if go is None:
            raise ImportError("Install plotly for interactive visualization")

        for scores in explanations.values():
            self._validate(tokens, scores)

        if normalize:
            explanations = {
                k: self._normalize(v) for k, v in explanations.items()
            }

        fig = go.Figure()

        for name, scores in explanations.items():
            fig.add_trace(
                go.Scatter(
                    x=tokens,
                    y=scores,
                    mode="lines+markers",
                    name=name,
                )
            )

        fig.update_layout(
            title="Interactive Explanation Visualization",
            xaxis_title="Tokens",
            yaxis_title="Importance",
        )

        fig.show()

    # =====================================================
    # TOKEN HEATMAP
    # =====================================================

    def plot_token_heatmap(
        self,
        tokens: List[str],
        scores: List[float],
        *,
        normalize: bool = True,
        title: str = "Token Importance Heatmap",
        save_path: Optional[str] = None,
    ):

        self._validate(tokens, scores)

        if normalize:
            scores = self._normalize(scores)

        matrix = np.array(scores).reshape(1, -1)

        import matplotlib.pyplot as plt  # GPU-5: lazy import
        fig, ax = plt.subplots(figsize=(max(len(tokens) * 0.5, 8), 2))

        im = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticks([])
        ax.set_title(title)

        fig.colorbar(im, ax=ax)
        plt.tight_layout()

        self._finalize(fig, save_path)

    # =====================================================
    # BAR CHART
    # =====================================================

    def plot_importance_bar(
        self,
        tokens: List[str],
        scores: List[float],
        *,
        top_k: int = 20,
        normalize: bool = True,
        save_path: Optional[str] = None,
    ):

        self._validate(tokens, scores)

        if normalize:
            scores = self._normalize(scores)

        tokens_arr = np.array(tokens)
        scores_arr = np.array(scores)

        order = np.argsort(scores_arr)[::-1][:top_k]

        import matplotlib.pyplot as plt  # GPU-5: lazy import
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(tokens_arr[order][::-1], scores_arr[order][::-1])

        ax.set_title("Top Token Importance")
        plt.tight_layout()

        self._finalize(fig, save_path)

    # =====================================================
    #  UPDATED FULL VIEW
    # =====================================================

    def visualize_aggregated(
        self,
        aggregated_output: Dict,
        *,
        method_outputs: Optional[Dict[str, List[float]]] = None,
        save_prefix: Optional[str] = None,
        interactive: bool = False,
    ):

        tokens = aggregated_output.get("tokens", [])
        importance = aggregated_output.get("final_token_importance", [])

        if not tokens or not importance:
            raise ValueError("Invalid aggregated output")

        # Base plots
        self.plot_token_heatmap(
            tokens,
            importance,
            save_path=f"{save_prefix}_heatmap.png" if save_prefix else None,
        )

        self.plot_importance_bar(
            tokens,
            importance,
            save_path=f"{save_prefix}_bar.png" if save_prefix else None,
        )

        # multi-method overlay
        if method_outputs:
            self.plot_multi_method_overlay(
                tokens,
                method_outputs,
                save_path=f"{save_prefix}_overlay.png" if save_prefix else None,
            )

            #  interactive
            if interactive:
                self.plot_interactive(tokens, method_outputs)