"""Uniform interface over the four candidate encoders.

Hides their differing pooling, prefix and instruction conventions, so an
encoder swap is a config change. Emits dense vectors and, where the model
supports it, sparse term weights.

Implemented by ticket 05.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..config import EncoderConfig


class Encoder(Protocol):
    """What every encoder adapter offers the rest of the pipeline."""

    @property
    def dimensions(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_labels(self, texts: Sequence[str]) -> np.ndarray: ...


def load(config: EncoderConfig, device: str) -> Encoder:
    """Build the adapter named by the config, on the given device."""
    raise NotImplementedError("ticket 05: encoder adapter")
