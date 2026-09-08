"""Which device the work runs on, and what to say when it cannot.

Indexing, retrieval, reranking and all evaluation run on Apple Silicon; only the
contrastive fine-tunes need CUDA. A run that requires the GPU host must say so
in one sentence naming the environment file, rather than failing somewhere
inside a model load.
"""

from __future__ import annotations

DEVICES = ("cuda", "mps", "cpu")

GPU_HOST_HINT = (
    "Run it on the GPU host (`ssh nlp2`) with an environment installed from "
    "requirements/gpu.txt; requirements/mac.txt is CUDA-free by design."
)


class HardwareUnavailable(RuntimeError):
    """The requested device is not present on this machine."""


def select_device(prefer: str = "auto", *, torch=None) -> str:
    """Resolve a device name, preferring cuda, then mps, then cpu.

    `prefer` may name a device explicitly, in which case its absence is an
    error rather than a silent downgrade — a fine-tune that quietly lands on
    cpu is a wasted day.
    """
    torch = torch if torch is not None else _import_torch()
    available = _available(torch)

    if prefer == "auto":
        return next(device for device in DEVICES if device in available)

    if prefer not in DEVICES:
        raise HardwareUnavailable(
            f"unknown device {prefer!r}; expected one of {', '.join(DEVICES)}"
        )
    if prefer not in available:
        raise HardwareUnavailable(
            f"device {prefer!r} is not available here (available: "
            f"{', '.join(sorted(available))}). {GPU_HOST_HINT}"
        )
    return prefer


def require_cuda(what: str, *, torch=None) -> str:
    """Return "cuda", or explain why `what` cannot run on this machine."""
    torch = torch if torch is not None else _import_torch()
    available = _available(torch)
    if "cuda" not in available:
        raise HardwareUnavailable(
            f"{what} requires CUDA, which is not available here (available: "
            f"{', '.join(sorted(available))}). {GPU_HOST_HINT}"
        )
    return "cuda"


def describe_device(device: str) -> str:
    """One line naming the device, for the top of a run's log."""
    detail = {
        "cuda": "CUDA GPU",
        "mps": "Apple Silicon GPU via Metal",
        "cpu": "CPU only, expect this to be slow",
    }[device]
    return f"device: {device} ({detail})"


def _available(torch) -> set[str]:
    available = {"cpu"}
    if torch.cuda.is_available():
        available.add("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        available.add("mps")
    return available


def _import_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment failure path
        raise HardwareUnavailable(
            "torch is not installed. Install an environment first: "
            "`uv pip install -r requirements/mac.txt` on Apple Silicon, "
            "`pip install -r requirements/gpu.txt` on the GPU host."
        ) from error
    return torch
