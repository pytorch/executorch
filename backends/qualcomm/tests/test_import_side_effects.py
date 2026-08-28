# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that importing the Qualcomm backend package has no side effects.

The package's __init__ used to do real work at import time. Merely importing it fetched the
SDK and libc++ over the network, could re-execute the interpreter under a staged loader,
rewrote QNN_SDK_ROOT and LD_LIBRARY_PATH, raised on a machine with no network, and disabled
PyTorch's MKLDNN backend for the whole process on any AMD host. None of that is needed to
load the package: the native adaptor resolves QNN symbols with dlopen when a model is
compiled, so setup belongs on the paths that compile.

These are unit tests because the behaviour under test is what happens during import, which
is observable without a Qualcomm SDK or a Qualcomm device.
"""

import sys
import types

import executorch.backends.qualcomm as qnn
import pytest
import torch


@pytest.fixture
def fake_cpuinfo(monkeypatch):
    """Replaces cpuinfo so a vendor can be chosen without an AMD host."""

    def install(vendor):
        module = types.ModuleType("cpuinfo")
        module.get_cpu_info = lambda: {"vendor_id_raw": vendor}
        monkeypatch.setitem(sys.modules, "cpuinfo", module)

    return install


def test_importing_the_package_leaves_mkldnn_alone(monkeypatch, fake_cpuinfo):
    """Import must not change a global PyTorch setting, even on an AMD host.

    The setting affects every model in the interpreter, not just a Qualcomm one, so an
    import that flips it changes how unrelated code runs.
    """
    fake_cpuinfo("AuthenticAMD")
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)

    importlib = pytest.importorskip("importlib")
    importlib.reload(qnn)

    assert torch.backends.mkldnn.enabled


def test_importing_the_package_does_not_set_up_the_sdk(monkeypatch):
    """Import must not run the installer, which downloads and rewrites the environment."""
    calls = []
    monkeypatch.setattr(qnn, "install_qnn_sdk", lambda: calls.append(1) or True)
    monkeypatch.setattr(qnn, "_sdk_ready", False)

    importlib = pytest.importorskip("importlib")
    importlib.reload(qnn)

    assert not calls


def test_setup_is_idempotent(monkeypatch):
    """The compile paths each call it, so only the first call may do the work."""
    calls = []
    monkeypatch.setattr(qnn, "install_qnn_sdk", lambda: calls.append(1) or True)
    monkeypatch.setattr(qnn, "is_linux_x86", lambda: True)
    monkeypatch.setattr(qnn, "_sdk_ready", False)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.delenv("EXECUTORCH_BUILDING_WHEEL", raising=False)

    qnn.setup_qnn_sdk()
    qnn.setup_qnn_sdk()

    assert len(calls) == 1


def test_setup_honours_a_preinstalled_sdk(monkeypatch):
    calls = []
    monkeypatch.setattr(qnn, "install_qnn_sdk", lambda: calls.append(1) or True)
    monkeypatch.setattr(qnn, "_sdk_ready", False)
    monkeypatch.setenv("QNN_SDK_ROOT", "/opt/qcom/sdk")

    qnn.setup_qnn_sdk()

    assert not calls


def test_setup_reports_a_failed_install(monkeypatch):
    """A failure has to name the two ways out, since it cannot be resolved automatically."""
    monkeypatch.setattr(qnn, "install_qnn_sdk", lambda: False)
    monkeypatch.setattr(qnn, "is_linux_x86", lambda: True)
    monkeypatch.setattr(qnn, "_sdk_ready", False)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.delenv("EXECUTORCH_BUILDING_WHEEL", raising=False)

    with pytest.raises(RuntimeError, match="QNN_SDK_ROOT"):
        qnn.setup_qnn_sdk()


@pytest.mark.parametrize(
    "vendor,expected",
    [
        # The vendor string an AMD host reports. MKLDNN produces wrong results there.
        ("AuthenticAMD", False),
        ("GenuineIntel", True),
        # A host that reports nothing, such as Apple silicon.
        ("", True),
    ],
)
def test_mkldnn_is_disabled_only_on_amd(monkeypatch, fake_cpuinfo, vendor, expected):
    fake_cpuinfo(vendor)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)

    qnn.disable_mkldnn_on_amd()

    assert torch.backends.mkldnn.enabled is expected
