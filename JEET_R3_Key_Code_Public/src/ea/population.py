"""Population utilities for EA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Individual:
    """EA individual with parameter vector and optional objective values."""

    theta: np.ndarray
    F: Optional[np.ndarray] = None  # shape (3,) cost objectives (minimize)
