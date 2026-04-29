"""Label generation for supervised learning targets."""

from trading.labels.forward_returns import compute_labels, label_columns_for_horizon
from trading.labels.store import LabelStore

__all__ = ["LabelStore", "compute_labels", "label_columns_for_horizon"]
