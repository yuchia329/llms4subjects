"""The environment split is a hard constraint, so it gets a test.

Everything except the three training runs happens on Apple Silicon; a CUDA
wheel reaching requirements/mac.txt would break the local loop silently, since
pip resolves it happily and only the import fails.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements"

CUDA_MARKERS = ("nvidia-", "+cu", "download.pytorch.org/whl/cu")


def read(name):
    return (REQUIREMENTS / name).read_text()


def test_the_two_environments_and_their_shared_base_exist():
    for name in ("base.txt", "mac.txt", "gpu.txt"):
        assert (REQUIREMENTS / name).is_file(), name


@pytest.mark.parametrize("name", ["base.txt", "mac.txt"])
def test_the_local_environment_names_no_cuda_wheel(name):
    text = read(name)
    for marker in CUDA_MARKERS:
        assert marker not in text, f"{name} pulls CUDA via {marker!r}"


def test_the_gpu_environment_pins_the_cuda_build():
    text = read("gpu.txt")
    assert "download.pytorch.org/whl/cu" in text
    assert "+cu" in text


@pytest.mark.parametrize("name", ["mac.txt", "gpu.txt"])
def test_each_environment_builds_on_the_shared_base(name):
    assert "-r base.txt" in read(name)


def test_the_torch_pin_lives_in_the_environment_files_not_the_base():
    """The torch build is the only pin that differs between the two hosts."""
    base = [line.split("#")[0].strip() for line in read("base.txt").splitlines()]
    assert not any(line.startswith("torch==") for line in base)
    assert "torch==2.5.1" in read("mac.txt")
    assert "torch==2.5.1+cu124" in read("gpu.txt")


def test_the_old_cuda_pinned_requirements_file_is_gone():
    """requirements.txt pinned nvidia wheels; it is replaced, not kept beside."""
    assert not (REPO_ROOT / "requirements.txt").exists()
