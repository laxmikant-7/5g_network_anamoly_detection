"""Test configuration."""

from __future__ import annotations

import sys

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
