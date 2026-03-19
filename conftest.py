from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POLYFOLDS_ROOT = ROOT / "polyfolds"
if str(POLYFOLDS_ROOT) not in sys.path:
    sys.path.insert(0, str(POLYFOLDS_ROOT))
