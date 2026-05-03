"""Model architectures package.

Houses the nn.Module classes used by the training and inference paths.
Kept separate from ``src.models.encoder`` / ``src.models.heads`` because
those are component libraries (encoders, classification heads), whereas
``architectures`` exposes end-to-end assembled models.
"""

from src.models.architectures.hybrid_truthlens_model import HybridTruthLensModel

__all__ = ["HybridTruthLensModel"]
