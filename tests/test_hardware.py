"""Hardware selection: a Mac must get a clear message, never an import error."""

import pytest

from llms4subjects.hardware import (
    HardwareUnavailable,
    describe_device,
    require_cuda,
    select_device,
)


class FakeTorch:
    def __init__(self, cuda=False, mps=False):
        self._cuda, self._mps = cuda, mps
        self.cuda = type("cuda", (), {"is_available": lambda _self: self._cuda})()
        self.backends = type(
            "backends",
            (),
            {"mps": type("mps", (), {"is_available": lambda _self: self._mps})()},
        )()


def test_prefers_cuda_then_mps_then_cpu():
    assert select_device(torch=FakeTorch(cuda=True, mps=True)) == "cuda"
    assert select_device(torch=FakeTorch(cuda=False, mps=True)) == "mps"
    assert select_device(torch=FakeTorch()) == "cpu"


def test_explicit_request_for_missing_hardware_fails_with_a_clear_message():
    with pytest.raises(HardwareUnavailable) as excinfo:
        select_device("cuda", torch=FakeTorch(mps=True))

    message = str(excinfo.value)
    assert "cuda" in message
    assert "requirements/gpu.txt" in message


def test_require_cuda_names_the_environment_file_to_install():
    with pytest.raises(HardwareUnavailable) as excinfo:
        require_cuda("contrastive fine-tuning", torch=FakeTorch(mps=True))

    message = str(excinfo.value)
    assert "contrastive fine-tuning" in message
    assert "requirements/gpu.txt" in message
    assert "mps" in message


def test_require_cuda_returns_the_device_when_present():
    assert require_cuda("training", torch=FakeTorch(cuda=True)) == "cuda"


def test_describe_device_is_human_readable():
    assert "mps" in describe_device("mps")


def test_unknown_device_name_is_refused():
    with pytest.raises(HardwareUnavailable, match="tpu"):
        select_device("tpu", torch=FakeTorch())


def test_real_torch_resolves_a_device():
    """On this machine, whatever it is, selection must not raise."""
    assert select_device() in {"cuda", "mps", "cpu"}
