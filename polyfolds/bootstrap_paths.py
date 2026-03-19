"""Shared path bootstrap for the Polyfolds offline workspace."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_polyfolds_paths() -> None:
    """Make the Polyfolds root and legacy helper modules importable."""

    root = Path(__file__).resolve().parent
    legacy = root / "legacy"
    for path in (root, legacy):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
