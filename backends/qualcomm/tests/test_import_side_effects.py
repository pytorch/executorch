# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that loading the Qualcomm backend has no side effects.

Setup used to run while the package's ``__init__`` was imported. Merely importing it fetched the
SDK and libc++ over the network, could re-execute the interpreter under a staged loader, rewrote
QNN_SDK_ROOT and LD_LIBRARY_PATH, raised on a machine with no network, and disabled PyTorch's
MKLDNN backend for the whole process on any AMD host. None of that is needed to load the
package: the native adaptor resolves QNN symbols with dlopen when a backend is started, so setup
belongs on the calls that start one.

Two separate properties are checked. First, that no module runs the setup while being imported,
which is asserted against the module source because that is what an import executes. Second,
that the setup itself behaves, which is asserted by calling it.

The source checks ask the module's loader for its source rather than opening a file, so they
also hold where the package was assembled into an archive, and a module with no source of its
own reads as empty instead of raising.
"""

import ast
import builtins
import os
import platform
import sys
import types

import executorch.backends.qualcomm as qnn
import executorch.backends.qualcomm.builders.node_visitor as node_visitor
import executorch.backends.qualcomm.debugger.utils as debugger_utils
import executorch.backends.qualcomm.quantizer.validators as validators
import executorch.backends.qualcomm.utils.check_qnn_version as check_qnn_version
import executorch.backends.qualcomm.utils.qnn_manager_lifecycle as qnn_manager_lifecycle
import executorch.backends.qualcomm.utils.qnn_sdk_setup as qnn_sdk_setup
import executorch.backends.qualcomm.utils.utils as qnn_utils
import pytest
import torch
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)

try:
    from executorch.backends.qualcomm.recipes.qnn_recipe_types import QNNRecipeType
except Exception:  # the recipe package pulls in optional dependencies
    QNNRecipeType = None

# Every module that reaches the SDK. Each one is imported by ordinary use of the backend, so a
# setup call in any of them is a setup call at import time.
CONSUMER_MODULES = [
    node_visitor,
    qnn_utils,
    check_qnn_version,
    qnn_manager_lifecycle,
    validators,
    debugger_utils,
]

SETUP_NAMES = {"setup_qnn_sdk", "disable_mkldnn_on_amd", "install_qnn_sdk"}


def module_source(module):
    """Returns `module`'s own source text, or an empty string when it has none.

    Read through the loader rather than from the file, so this also works where the package was
    assembled into an archive and no file exists to open. A loader is allowed to report missing
    source either by returning None or by raising ImportError, and a build system that
    synthesizes an empty file leaves a module with no source of its own, so both have to read as
    empty here rather than failing the check.
    """
    loader = getattr(module, "__loader__", None)
    get_source = getattr(loader, "get_source", None)
    if get_source is None:
        return ""
    try:
        return get_source(module.__name__) or ""
    except ImportError:
        return ""


def calls_made_while_importing(module):
    """Returns the names called by `module`'s own top level statements.

    Walks into every top level statement rather than only bare expressions, because a call
    guarded by an `if` still runs during the import. That guarded shape is what the original
    defect looked like, so a check that skipped it would not have caught it.

    A function body is skipped, since an import does not enter it, but the parts of the definition
    around it are not: a decorator, a base class and a default argument are all evaluated at import
    time. A version check inside a `skipIf` decorator is exactly how setup crept back onto the
    import path once already. A class body is not skipped, because unlike a function body it does
    run while the module is imported.
    """
    tree = ast.parse(module_source(module))
    called = set()

    def record(nodes):
        for root in nodes:
            for node in ast.walk(root):
                if isinstance(node, ast.Call):
                    target = node.func
                    if isinstance(target, ast.Name):
                        called.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        called.add(target.attr)

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record(statement.decorator_list)
            if isinstance(statement, ast.ClassDef):
                record(statement.bases)
                record(kw.value for kw in statement.keywords)
                # A class body IS executed while the module is imported, unlike a function body,
                # so anything called directly in one has to be seen here. Its methods are skipped
                # for the same reason a top-level function is, except for their decorators.
                for member in statement.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        record(member.decorator_list)
                    else:
                        record([member])
            else:
                args = statement.args
                record(d for d in args.defaults if d is not None)
                record(d for d in args.kw_defaults if d is not None)
            continue
        record([statement])
    return called


@pytest.fixture
def fake_cpuinfo(monkeypatch):
    """Replaces cpuinfo so a vendor can be chosen without an AMD host."""

    def install(vendor, key="vendor_id_raw"):
        module = types.ModuleType("cpuinfo")
        module.get_cpu_info = lambda: {key: vendor}
        monkeypatch.setitem(sys.modules, "cpuinfo", module)

    return install


@pytest.fixture
def ready_to_set_up(monkeypatch):
    """Puts the setup back in its pre-run state, on a host that could download an SDK."""
    monkeypatch.setattr(qnn_sdk_setup, "_sdk_ready", False)
    monkeypatch.setattr(qnn_sdk_setup, "_is_linux_x86", lambda: True)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.delenv("EXECUTORCH_BUILDING_WHEEL", raising=False)


@pytest.fixture
def recorded_install(monkeypatch):
    """Replaces the installer with one that records its calls instead of downloading."""
    calls = []

    def install():
        calls.append(1)
        return True

    monkeypatch.setattr(qnn_sdk_setup, "_install_qnn_sdk", install)
    return calls


@pytest.mark.parametrize(
    "module", CONSUMER_MODULES, ids=lambda module: module.__name__.rsplit(".", 1)[-1]
)
def test_importing_a_module_does_not_run_the_setup(module):
    """No module may set up the SDK while it is being imported.

    Asserted against the source rather than by importing with a stub in place: by the time a test
    runs, the import has already happened, and re-importing rebinds the real functions and
    discards any stub. What matters is that no top level statement calls them.
    """
    assert not calls_made_while_importing(module) & SETUP_NAMES


def test_the_package_root_stays_inert():
    """The package's __init__ must hold nothing a submodule needs.

    A build system that assembles a package from a file list can leave this file out and put an
    empty one in its place, which silently removes anything defined here. Keeping it free of top
    level statements means the real file and a synthesized empty one behave the same.
    """
    assert not ast.parse(module_source(qnn)).body


@pytest.mark.parametrize(
    "source",
    [
        "setup_qnn_sdk()\n",
        "if True:\n    setup_qnn_sdk()\n",
        "@skip_if(setup_qnn_sdk())\ndef f():\n    pass\n",
        "class C(base(setup_qnn_sdk())):\n    pass\n",
        "def f(x=setup_qnn_sdk()):\n    pass\n",
        "class C:\n    setup_qnn_sdk()\n",
        "class C:\n    @skip_if(setup_qnn_sdk())\n    def m(self):\n        pass\n",
    ],
    ids=[
        "plain",
        "guarded",
        "decorator",
        "class-base",
        "default-arg",
        "class-body",
        "method-decorator",
    ],
)
def test_the_check_sees_every_shape_that_runs_at_import(source):
    """Guards the two checks above, which pass trivially if the walk finds nothing.

    All seven shapes run while a module is imported. Four of them are easy to miss: three sit on a
    function or class definition, whose body an import does not enter, and the class body itself
    does run even though a function body does not. A version check inside a decorator is how setup
    crept back onto the import path once already.
    """
    module = types.ModuleType("shaped_setup_call")
    module.__loader__ = types.SimpleNamespace(get_source=lambda name: source)

    assert calls_made_while_importing(module) & SETUP_NAMES


def test_the_check_treats_a_sourceless_module_as_empty():
    """A synthesized empty file has no source, and that has to read as empty, not raise."""
    module = types.ModuleType("synthesized_package")
    module.__loader__ = types.SimpleNamespace(get_source=lambda name: None)

    assert module_source(module) == ""
    assert not calls_made_while_importing(module)


def test_the_check_treats_a_refusing_loader_as_empty():
    """A loader may report missing source by raising instead of returning None."""

    def refuse(name):
        raise ImportError(f"no source available for {name}")

    module = types.ModuleType("bytecode_only_package")
    module.__loader__ = types.SimpleNamespace(get_source=refuse)

    assert module_source(module) == ""
    assert not calls_made_while_importing(module)


class _StopAfterTrace(Exception):
    """Ends an entry point at the trace, so a test can check what ran before it."""


@pytest.mark.parametrize(
    "entry_point", ["to_edge_transform_and_lower_to_qnn", "capture_program"]
)
def test_setup_and_the_amd_guard_precede_the_trace(monkeypatch, entry_point):
    """Both have to happen at the entry point, not deeper in the lowering.

    The SDK setup can download one, and on an old glibc the installer re-executes the
    interpreter, so running it after a trace would throw the traced model away. The AMD guard
    prevents a crash that needs a real convolution, which calibration and a plain forward pass
    run even though a trace does not.

    Driven by recording the real call order rather than by comparing source positions, because a
    call inside a function that is never called would satisfy a source order check.
    """
    order = []
    monkeypatch.setattr(
        qnn_utils, "disable_mkldnn_on_amd", lambda: order.append("guard")
    )
    monkeypatch.setattr(qnn_utils, "setup_qnn_sdk", lambda: order.append("setup"))

    def stop_here(*args, **kwargs):
        order.append("trace")
        raise _StopAfterTrace("stopped at the trace")

    monkeypatch.setattr(torch.export, "export", stop_here)

    # The entry point is expected to fail on the stub inputs, one way or another. What matters is
    # that both have already run by then. Setup can re-execute the interpreter, which would throw
    # away a traced model, and the guard has to precede any eager run.
    with pytest.raises((_StopAfterTrace, ValueError, TypeError, AttributeError)):
        getattr(qnn_utils, entry_point)(torch.nn.Identity(), (torch.ones(1),), [])

    assert order, f"{entry_point} applied neither the setup nor the AMD guard"
    assert "setup" in order, f"{entry_point} does not set up the SDK before tracing"
    assert "guard" in order, f"{entry_point} does not apply the AMD guard"
    if "trace" in order:
        assert order.index("setup") < order.index("trace")
        assert order.index("guard") < order.index("trace")


def test_the_op_info_is_looked_for_again_after_setup(monkeypatch):
    """The import-time attempt can run before anything has made the SDK usable.

    Drives the real retry, rather than stubbing it, so a version that only trusts the
    import-time result fails here.
    """
    from executorch.backends.qualcomm.quantizer import backend_opinfo_adapter as adapter

    # As if the import-time attempt had failed, which is the case this exists for.
    monkeypatch.setattr(adapter, "_HAS_BACKEND_OPINFO", False)
    monkeypatch.setattr(adapter, "backend_opinfo", None)

    staged = types.ModuleType("qti.aisw.converters.common")
    staged.backend_opinfo = types.SimpleNamespace(
        HTP=1, BackendOpInfo=lambda *a: "real"
    )

    def stage_the_sdk():
        # What setup does for real: makes the SDK importable.
        monkeypatch.setitem(sys.modules, "qti", types.ModuleType("qti"))
        monkeypatch.setitem(sys.modules, "qti.aisw", types.ModuleType("qti.aisw"))
        monkeypatch.setitem(
            sys.modules, "qti.aisw.converters", types.ModuleType("qti.aisw.converters")
        )
        monkeypatch.setitem(sys.modules, "qti.aisw.converters.common", staged)

    # Patched at its source, because the retry imports it inside the function.
    monkeypatch.setattr(qnn_sdk_setup, "setup_qnn_sdk", lambda: None)
    monkeypatch.setattr(adapter, "add_qnn_python_path", stage_the_sdk)

    assert adapter._load_backend_opinfo() is True
    assert adapter._HAS_BACKEND_OPINFO


def test_the_op_info_fallback_is_not_cached_forever(monkeypatch):
    """A do-nothing checker must not be pinned once the SDK becomes usable.

    The real getter is cached, so caching the fallback too would keep returning it for those
    arguments for the rest of the process, silently dropping every constraint check.
    """
    from executorch.backends.qualcomm.quantizer import backend_opinfo_adapter as adapter

    ready = []
    monkeypatch.setattr(adapter, "_load_backend_opinfo", lambda: bool(ready))
    monkeypatch.setattr(adapter, "_warn_once_about_the_fallback", lambda: None)
    monkeypatch.setattr(
        adapter, "_get_backend_opinfo_cached", lambda backend, soc: "real"
    )

    before = adapter.get_backend_opinfo("HTP", 1)
    ready.append(True)
    after = adapter.get_backend_opinfo("HTP", 1)

    assert isinstance(before, adapter._NoOpBackendOpInfo)
    assert after == "real"


@pytest.mark.parametrize(
    "helper,unset_answer",
    [
        ("is_qnn_sdk_version_less_than", True),
        ("is_qnn_sdk_version_greater_than", False),
    ],
)
def test_a_broken_sdk_is_not_reported_as_an_old_one(monkeypatch, helper, unset_answer):
    """An unreadable SDK must not look like an old SDK.

    With no SDK path there is no version to compare, so a fallback answer is right. A library
    that is present but cannot be read is a different problem, and swallowing it too made the
    two indistinguishable, so a real failure was silently reported as a version gate.
    """

    def raise_it(error):
        def raiser():
            raise error

        return raiser

    no_path = check_qnn_version.QnnSdkRootNotSet("QNN_SDK_ROOT must be set.")
    monkeypatch.setattr(check_qnn_version, "get_sdk_build_id", raise_it(no_path))

    assert getattr(check_qnn_version, helper)("2.48") is unset_answer

    unreadable = OSError("cannot open libQnnHtp.so")
    monkeypatch.setattr(check_qnn_version, "get_sdk_build_id", raise_it(unreadable))

    with pytest.raises(OSError):
        getattr(check_qnn_version, helper)("2.48")


def test_reusing_a_cached_manager_still_applies_the_guard(
    monkeypatch, fake_cpuinfo, recorded_install
):
    """A second lowering must not run with the setting back on.

    The manager is built once and reused, so applying the guard only where it is built would let
    every later lowering run in exactly the configuration the guard exists to prevent.
    """
    fake_cpuinfo("AuthenticAMD")
    monkeypatch.setattr(qnn_sdk_setup, "_vendor_is_amd", None)
    # Marked ready, because this test is about the guard and nothing here may reach the real
    # installer: it downloads about a gigabyte and rewrites the environment for the whole process.
    monkeypatch.setattr(qnn_sdk_setup, "_sdk_ready", True)
    registry = qnn_manager_lifecycle.QnnManagerRegistry()
    backend_type = QnnExecuTorchBackendType.kHtpBackend
    registry._registry[backend_type] = object()

    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)
    registry.get_or_create_qnn_manager(backend_type, b"")

    assert not torch.backends.mkldnn.enabled
    # Nothing may reach the real installer from a test: it downloads about a gigabyte and
    # rewrites the environment for the rest of the process.
    assert not recorded_install


@pytest.mark.parametrize(
    "module_name,attribute,call",
    [
        ("qnn_utils", "update_spill_fill_size", lambda f: f([])),
        (
            "export_utils",
            "QnnConfig",
            lambda f: f(soc_model="SM8650", build_folder="/tmp/bf"),
        ),
        (
            "target_recipes",
            "get_android_recipe",
            lambda f: f("android-arm64-snapdragon-fp16"),
        ),
        ("qnn_utils", "from_context_binary", lambda f: f("nope.bin", "g")),
        ("qnn_utils", "skip_annotation", lambda f: f(None, None, [], (), None)),
        (
            "recipe_provider",
            "QNNRecipeProvider",
            lambda f: f().create_recipe(
                QNNRecipeType.FP16 if QNNRecipeType else "qnn_fp16",
                soc_model="SM8650",
            ),
        ),
        ("cli", "execute", lambda f: f(None)),
    ],
)
def test_every_entry_point_reaches_the_setup(monkeypatch, module_name, attribute, call):
    """Each of these needs a usable SDK, and none of them is a lowering entry point.

    Without a test per call site, deleting one is invisible: the whole premise is that setup no
    longer happens at import, so a missing call is a broken path rather than a slower one.
    """
    # Imported here rather than at module scope, so one missing optional dependency cannot stop
    # the whole file from being collected.
    paths = {
        "qnn_utils": "executorch.backends.qualcomm.utils.utils",
        "export_utils": "executorch.backends.qualcomm.export_utils",
        "target_recipes": "executorch.export.target_recipes",
        "quantizer": "executorch.backends.qualcomm.quantizer.quantizer",
        "recipe_provider": "executorch.backends.qualcomm.recipes.qnn_recipe_provider",
        "cli": "executorch.examples.qualcomm.util_scripts.cli",
        "qaihub_export": "executorch.examples.qualcomm.qaihub_scripts.utils.export",
    }
    module = pytest.importorskip(paths[module_name])
    reached = set()
    # Patched at the source module too, because some call sites import the helpers inside the
    # function, where rebinding the caller's attribute would miss them.
    for name in ("setup_qnn_sdk", "disable_mkldnn_on_amd"):
        recorder = (lambda n: lambda *a, **k: reached.add(n))(name)
        monkeypatch.setattr(qnn_sdk_setup, name, recorder)
        if hasattr(module, name):
            monkeypatch.setattr(module, name, recorder)
    # Linux x86 only on some paths, so the platform gate must not short-circuit the probe. Patched
    # where each module looks it up, since they import the platform module separately.
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    export_utils = pytest.importorskip("executorch.export.utils")
    monkeypatch.setattr(
        export_utils, "is_supported_platform_for_qnn_lowering", lambda: True
    )
    if hasattr(module, "is_supported_platform_for_qnn_lowering"):
        monkeypatch.setattr(
            module, "is_supported_platform_for_qnn_lowering", lambda: True
        )

    # Every one of these fails on the stub input. What matters is what ran before it did.
    try:
        call(getattr(module, attribute))
    except Exception as error:  # noqa: BLE001
        # Matched on the registry's own wording rather than on "not available" alone, and only
        # when the backend is genuinely absent. A deleted setup call also leaves `reached` empty
        # and can reach a wrapped message containing those same words, so the looser condition
        # turned the regression this test exists to catch into a green skip.
        unavailable = "Backend 'qnn' not available" in str(error)
        if not reached and unavailable:
            pytest.skip(f"{module_name} is not usable in this environment: {error}")

    # Named rather than "either helper ran", because most of these call sites apply the AMD guard
    # as well, and one shared flag lets that one hide a deleted setup call.
    assert (
        "setup_qnn_sdk" in reached
    ), f"{module_name}.{attribute} never reaches the setup"


def test_a_failed_install_does_not_leave_a_broken_sdk_path(
    monkeypatch, ready_to_set_up
):
    """A failed install must not look like a usable SDK to the next caller.

    The installer prepends the staged library directory to LD_LIBRARY_PATH before it decides
    whether the SDK is usable, and it reads that variable back on its next call: the QNN libraries
    being findable through it count as proof that a usable SDK is already present. So a failure
    that leaves the library path behind is the worse half. The next call takes that shortcut and
    reports success with no SDK path set at all.
    """
    staged_libs = "/staged/but/broken/lib/x86_64-linux-clang"
    attempts = []

    def poisoning_install():
        os.environ["QNN_SDK_ROOT"] = "/staged/but/broken"
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{staged_libs}:{ld_path}" if ld_path else staged_libs
        )
        attempts.append(1)
        return False

    monkeypatch.setattr(qnn_sdk_setup, "_install_qnn_sdk", poisoning_install)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unrelated/lib")

    with pytest.raises(RuntimeError):
        qnn_sdk_setup.setup_qnn_sdk()

    assert os.environ.get("QNN_SDK_ROOT") is None
    assert os.environ.get("LD_LIBRARY_PATH") == "/unrelated/lib"

    # The docstring promises the next caller tries again, which only holds if both were restored.
    with pytest.raises(RuntimeError):
        qnn_sdk_setup.setup_qnn_sdk()

    assert len(attempts) == 2


def test_a_failed_install_restores_an_absent_library_path(monkeypatch, ready_to_set_up):
    """Restoring must remove the variable the installer created, not set it to an empty string.

    The installer creates the variable from nothing when the process started without it, and an
    empty string is inherited by every child process the example scripts launch.
    """
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    def poisoning_install():
        os.environ["QNN_SDK_ROOT"] = "/staged/but/broken"
        os.environ["LD_LIBRARY_PATH"] = "/staged/but/broken/lib/x86_64-linux-clang"
        return False

    monkeypatch.setattr(qnn_sdk_setup, "_install_qnn_sdk", poisoning_install)

    with pytest.raises(RuntimeError):
        qnn_sdk_setup.setup_qnn_sdk()

    assert "LD_LIBRARY_PATH" not in os.environ


def test_an_installer_that_raises_also_leaves_nothing_behind(
    monkeypatch, ready_to_set_up
):
    """The installer raises as well as returning False, and both need the same cleanup.

    A missing unpacking tool and an unrecognised archive both escape it, after the point where it
    has already written the two variables.
    """
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unrelated/lib")

    def raising_install():
        os.environ["QNN_SDK_ROOT"] = "/staged/but/broken"
        os.environ["LD_LIBRARY_PATH"] = "/staged/but/broken/lib:/unrelated/lib"
        raise ValueError("Unsupported archive format")

    monkeypatch.setattr(qnn_sdk_setup, "_install_qnn_sdk", raising_install)

    with pytest.raises(ValueError):
        qnn_sdk_setup.setup_qnn_sdk()

    assert os.environ.get("QNN_SDK_ROOT") is None
    assert os.environ.get("LD_LIBRARY_PATH") == "/unrelated/lib"


def test_building_a_quantizer_applies_the_amd_guard(monkeypatch, fake_cpuinfo):
    """Everything that calibrates builds a quantizer first, so the guard belongs there.

    The standalone quantize command in the example CLI is the case that made this matter: it runs
    calibration itself, and calibration is a real eager run of the model, which is what the guard
    is for. It reaches this constructor through make_quantizer.
    """
    fake_cpuinfo("AuthenticAMD")
    monkeypatch.setattr(qnn_sdk_setup, "_vendor_is_amd", None)
    monkeypatch.setattr(qnn_sdk_setup, "_sdk_ready", True)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)
    quantizer_module = pytest.importorskip(
        "executorch.backends.qualcomm.quantizer.quantizer"
    )

    quantizer_module.QnnQuantizer()

    assert not torch.backends.mkldnn.enabled


def test_a_missing_cpuinfo_does_not_stop_a_lowering(monkeypatch):
    """The AMD guard runs on every lowering, but it only matters on an AMD host.

    Raising when py-cpuinfo is absent would fail a lowering on an Intel or ARM machine over a
    dependency that machine has no use for.
    """
    monkeypatch.setattr(qnn_sdk_setup, "_vendor_is_amd", None)
    monkeypatch.setitem(sys.modules, "cpuinfo", None)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)

    qnn_sdk_setup.disable_mkldnn_on_amd()

    assert torch.backends.mkldnn.enabled


def test_setup_is_idempotent(ready_to_set_up, recorded_install):
    """Several call sites each ask for it, so only the first may do the work."""
    qnn_sdk_setup.setup_qnn_sdk()
    qnn_sdk_setup.setup_qnn_sdk()

    assert len(recorded_install) == 1


def test_setup_honours_a_preinstalled_sdk(
    monkeypatch, ready_to_set_up, recorded_install
):
    monkeypatch.setenv("QNN_SDK_ROOT", "/opt/qcom/sdk")

    qnn_sdk_setup.setup_qnn_sdk()

    assert not recorded_install


def test_setup_leaves_an_empty_sdk_root_alone(
    monkeypatch, ready_to_set_up, recorded_install
):
    """Setting the variable at all means something else manages the SDK.

    An empty value used to read as "not set" and fall through to the installer, which downloads
    an SDK and rewrites the environment underneath whatever set it.
    """
    monkeypatch.setenv("QNN_SDK_ROOT", "")

    qnn_sdk_setup.setup_qnn_sdk()

    assert not recorded_install


def test_setup_skips_a_wheel_build(monkeypatch, ready_to_set_up, recorded_install):
    monkeypatch.setenv("EXECUTORCH_BUILDING_WHEEL", "1")

    qnn_sdk_setup.setup_qnn_sdk()

    assert not recorded_install


def test_setup_skips_a_platform_with_no_published_sdk(
    monkeypatch, ready_to_set_up, recorded_install
):
    monkeypatch.setattr(qnn_sdk_setup, "_is_linux_x86", lambda: False)

    qnn_sdk_setup.setup_qnn_sdk()

    assert not recorded_install


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Linux", "x86_64", True),
        ("Linux", "AMD64", True),
        ("Linux", "i686", True),
        ("Linux", "aarch64", False),
        ("Darwin", "arm64", False),
        ("Darwin", "x86_64", False),
        ("Windows", "AMD64", False),
    ],
)
def test_the_platform_check_reads_the_real_platform(
    monkeypatch, system, machine, expected
):
    """Drives the real function, so its machine list is actually covered.

    Patching the check itself away would leave its body unexecuted, and the case it exists to
    prevent, reaching for the downloader just to name the platform, would ship unnoticed.
    """
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    # Absent downloader, so answering at all proves the check does not consult it.
    monkeypatch.setitem(
        sys.modules, "executorch.backends.qualcomm.scripts.download_qnn_sdk", None
    )

    assert qnn_sdk_setup._is_linux_x86() is expected


def test_a_missing_downloader_does_not_stop_a_lowering(monkeypatch, ready_to_set_up):
    """A build that leaves the downloader out still has to be able to lower.

    Such a build supplies the QNN libraries through its own build rules rather than by fetching
    them, so there is nothing for setup to do and nothing to report. Raising here fails a lowering
    that would otherwise work, and a library that really is missing still says so when the backend
    starts.
    """

    def no_downloader(name, *args, **kwargs):
        if name.startswith("executorch.backends.qualcomm.scripts"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", no_downloader)

    qnn_sdk_setup.setup_qnn_sdk()

    assert qnn_sdk_setup._sdk_ready
    # Nothing may be invented for a caller who never set it, since everything downstream builds
    # real paths from this value.
    assert os.environ.get("QNN_SDK_ROOT") is None


def test_a_dependency_missing_inside_the_downloader_speaks_for_itself(
    monkeypatch, ready_to_set_up
):
    """Only an absent downloader is reworded. Its own missing dependency must not be hidden.

    Reporting a missing `requests` as "set QNN_SDK_ROOT" would send the reader after the wrong
    problem.
    """

    def no_requests(name, *args, **kwargs):
        if name == "requests":
            raise ModuleNotFoundError("No module named 'requests'", name="requests")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    # Dropped from the module cache so the import below really runs and really fails on
    # `requests`. A cached downloader would sail past the block and call the real installer,
    # because the fixture has already cleared QNN_SDK_ROOT and forced the platform check true.
    monkeypatch.delitem(
        sys.modules,
        "executorch.backends.qualcomm.scripts.download_qnn_sdk",
        raising=False,
    )
    monkeypatch.setattr(builtins, "__import__", no_requests)

    with pytest.raises(ModuleNotFoundError, match="requests"):
        qnn_sdk_setup.setup_qnn_sdk()


def test_setup_runs_once_under_concurrent_callers(monkeypatch, ready_to_set_up):
    """Several call sites reach it, so two threads can arrive at the same time.

    Without a lock they both see the flag unset and both run the installer, which downloads an
    SDK and rewrites the environment.
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

    monkeypatch.setattr(qnn_sdk_setup, "_install_qnn_sdk", slow_install)

    threads = [threading.Thread(target=qnn_sdk_setup.setup_qnn_sdk) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    # Reported before the count, so a hung thread says so rather than showing a confusing
    # number of installer calls.
    assert not any(thread.is_alive() for thread in threads)
    assert len(calls) == 1


def test_setup_reports_a_failed_install(monkeypatch, ready_to_set_up):
    """A failure has to name the two ways out, since it cannot be resolved automatically."""
    monkeypatch.setattr(qnn_sdk_setup, "_install_qnn_sdk", lambda: False)

    # Both ways out are asserted, so a message trimmed down to the variable name fails here.
    with pytest.raises(RuntimeError, match="Download the SDK manually"):
        qnn_sdk_setup.setup_qnn_sdk()

    with pytest.raises(RuntimeError, match=r"export QNN_SDK_ROOT="):
        qnn_sdk_setup.setup_qnn_sdk()


@pytest.mark.parametrize("key", ["vendor_id_raw", "vendor_id"])
@pytest.mark.parametrize(
    "vendor,expected",
    [
        # The vendor string an AMD host reports. MKLDNN crashes there.
        ("AuthenticAMD", False),
        ("GenuineIntel", True),
        # A host that reports nothing, such as Apple silicon.
        ("", True),
    ],
)
def test_mkldnn_is_disabled_only_on_amd(
    monkeypatch, fake_cpuinfo, key, vendor, expected
):
    """Releases of py-cpuinfo disagree on the key, so both spellings have to be read."""
    fake_cpuinfo(vendor, key=key)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)
    monkeypatch.setattr(qnn_sdk_setup, "_vendor_is_amd", None)

    qnn_sdk_setup.disable_mkldnn_on_amd()

    assert torch.backends.mkldnn.enabled == expected


def test_the_vendor_is_read_at_most_once(monkeypatch):
    """Reading it costs a subprocess, and the call sites reach it more than once per lowering.

    py-cpuinfo caches nothing, so without a cache here every lowering paid for the read several
    times over, which was enough to push a delegate test job into its timeout.
    """
    reads = []
    module = types.ModuleType("cpuinfo")
    module.get_cpu_info = lambda: reads.append(1) or {"vendor_id_raw": "GenuineIntel"}
    monkeypatch.setitem(sys.modules, "cpuinfo", module)
    monkeypatch.setattr(qnn_sdk_setup, "_vendor_is_amd", None)

    for _ in range(5):
        qnn_sdk_setup.disable_mkldnn_on_amd()

    assert len(reads) == 1


def test_the_guard_holds_when_a_caller_re_enables_mkldnn(monkeypatch, fake_cpuinfo):
    """A later lowering must not run with the setting back on.

    Caching the decision rather than the vendor would leave the second lowering in exactly the
    configuration the guard exists to prevent, because the guard would return early.
    """
    reads = []
    module = types.ModuleType("cpuinfo")
    module.get_cpu_info = lambda: reads.append(1) or {"vendor_id_raw": "AuthenticAMD"}
    monkeypatch.setitem(sys.modules, "cpuinfo", module)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)
    monkeypatch.setattr(qnn_sdk_setup, "_vendor_is_amd", None)

    qnn_sdk_setup.disable_mkldnn_on_amd()
    assert not torch.backends.mkldnn.enabled

    torch.backends.mkldnn.enabled = True
    qnn_sdk_setup.disable_mkldnn_on_amd()

    assert not torch.backends.mkldnn.enabled
    # Re-applied without paying for the vendor read again.
    assert len(reads) == 1
