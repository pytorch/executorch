import importlib

import pytest


@pytest.fixture
def pin():
    """Yield a fresh import of torch_pin so tests can mutate CHANNEL safely."""
    import torch_pin

    yield torch_pin
    importlib.reload(torch_pin)


@pytest.mark.parametrize(
    "channel, expected_torch, expected_url",
    [
        (
            "nightly",
            "torch=={TORCH_VERSION}.{NIGHTLY_VERSION}",
            "https://download.pytorch.org/whl/nightly",
        ),
        ("test", "torch=={TORCH_VERSION}", "https://download.pytorch.org/whl/test"),
        ("release", "torch=={TORCH_VERSION}", "https://download.pytorch.org/whl"),
    ],
)
def test_channel_resolution(pin, channel, expected_torch, expected_url):
    pin.CHANNEL = channel
    expected = expected_torch.format(
        TORCH_VERSION=pin.TORCH_VERSION, NIGHTLY_VERSION=pin.NIGHTLY_VERSION
    )
    assert pin.torch_spec() == expected
    assert pin.torch_index_url_base() == expected_url


def test_all_specs_share_nightly_suffix(pin):
    pin.CHANNEL = "nightly"
    suffix = f".{pin.NIGHTLY_VERSION}"
    assert pin.torch_spec().endswith(suffix)
    assert pin.torchaudio_spec().endswith(suffix)
    assert pin.torchcodec_spec().endswith(suffix)
    assert pin.torchvision_spec().endswith(suffix)


def test_specs_drop_suffix_off_nightly(pin):
    pin.CHANNEL = "test"
    assert pin.torch_spec() == f"torch=={pin.TORCH_VERSION}"
    assert pin.torchaudio_spec() == f"torchaudio=={pin.TORCHAUDIO_VERSION}"
    assert pin.torchcodec_spec() == f"torchcodec=={pin.TORCHCODEC_VERSION}"
    assert pin.torchvision_spec() == f"torchvision=={pin.TORCHVISION_VERSION}"


def test_torch_branch_derived_from_version(pin):
    assert pin.torch_branch() == f"release/{pin.TORCH_VERSION.rsplit('.', 1)[0]}"
