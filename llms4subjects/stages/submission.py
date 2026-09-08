"""Writes the organizers' `<Type>/<lang>/<id>.json` tree, 50 ranked codes each.

This is the one stage that touches the filesystem as its purpose: its output is
the input to the official scorer, so the layout is the contract.

Implemented by ticket 03.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ..contracts import CODES_PER_RECORD, Code, Prediction


def write_submission(
    predictions: Sequence[Prediction],
    cells: Mapping[str, tuple[str, str]],
    destination: str | Path,
) -> list[Path]:
    """Write one file per record, returning the paths written."""
    raise NotImplementedError("ticket 03: submission writer")
