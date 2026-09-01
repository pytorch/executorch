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

import ast
import importlib
import sys
import types
from pathlib import Path

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

    The setting affects every model in the interpreter, not just a Qualcomm one, so an import
    that flips it changes how unrelated code runs.

    A reload is used deliberately: it re-executes every module level statement, which is exactly
    what an import does, and the stubbed cpuinfo is read inside those statements rather than
    bound at import, so unlike a stubbed installer it survives.
    """
    fake_cpuinfo("AuthenticAMD")
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)

    importlib.reload(qnn)

    assert torch.backends.mkldnn.enabled


def test_importing_the_package_does_not_set_up_the_sdk():
    """Import must not run the installer, which downloads and rewrites the environment.

    Asserted against the module source rather than by importing with a stub in place. A reload
    re-executes the module, which rebinds the real installer and discards any stub, so a stubbed
    reload can only ever observe an empty call list and would pass even if the call came back.
    What actually matters is that no module level statement calls it.
    """
    module = ast.parse(Path(qnn.__file__).read_text())
    called_at_module_level = {
        node.func.id
        for statement in module.body
        if isinstance(statement, ast.Expr)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "install_qnn_sdk" not in called_at_module_level
    assert "setup_qnn_sdk" not in called_at_module_level
    # A guard on the two above, which would pass just as happily if the parse found nothing.
    assert any(
        isinstance(statement, ast.FunctionDef) and statement.name == "setup_qnn_sdk"
        for statement in module.body
    )


def test_importing_the_package_does_not_need_the_downloader(monkeypatch):
    """Importing must not require the sibling scripts directory.

    Some builds package this backend without it, and a module level import of the downloader
    made every module that calls setup_qnn_sdk fail to import there with
    ModuleNotFoundError. Nothing about loading the package needs it: it is only used to
    download an SDK, which is one branch of setup.
    """
    monkeypatch.setitem(
        sys.modules, "executorch.backends.qualcomm.scripts.download_qnn_sdk", None
    )
    monkeypatch.setitem(sys.modules, "executorch.backends.qualcomm.scripts", None)

    importlib.reload(qnn)

    # The case that actually broke: a Linux x86 host with a preinstalled SDK, where setup runs
    # its early return. That must not touch the downloader.
    monkeypatch.setattr(qnn, "is_linux_x86", lambda: True)
    monkeypatch.setattr(qnn, "_sdk_ready", False)
    monkeypatch.setenv("QNN_SDK_ROOT", "/opt/qcom/sdk")
    monkeypatch.delenv("EXECUTORCH_BUILDING_WHEEL", raising=False)

    qnn.setup_qnn_sdk()

    assert qnn.disable_mkldnn_on_amd is not None


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


def test_setup_runs_once_under_concurrent_callers(monkeypatch):
    """Several modules call it, so two threads can reach it at the same time.

    Without a lock they both see the flag unset and both run the installer, which downloads an
    SDK and rewrites the environment. The import lock does not cover this, because the calls
    come from different modules rather than from one package __init__.
    """
    import threading
    import time

    calls = []

    def slow_install():
        # Widen the window between the check and the flag being set, so an unlocked version
        # fails reliably rather than occasionally.
        time.sleep(0.2)
        calls.append(1)
        return True

    monkeypatch.setattr(qnn, "install_qnn_sdk", slow_install)
    monkeypatch.setattr(qnn, "is_linux_x86", lambda: True)
    monkeypatch.setattr(qnn, "_sdk_ready", False)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.delenv("EXECUTORCH_BUILDING_WHEEL", raising=False)

    threads = [threading.Thread(target=qnn.setup_qnn_sdk) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    # Reported before the count, so a hung thread says so rather than showing a confusing
    # number of installer calls.
    assert not any(thread.is_alive() for thread in threads)
    assert len(calls) == 1


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
