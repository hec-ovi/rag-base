"""Make ``server`` importable when pytest is run from the repo root."""
from __future__ import annotations

import pathlib
import sys

SIDECAR_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(SIDECAR_DIR) not in sys.path:
    sys.path.insert(0, str(SIDECAR_DIR))
