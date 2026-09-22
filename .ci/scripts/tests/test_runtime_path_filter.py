# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the runtime search path filter packaging applies to shipped libraries.

Here rather than in the wheel checks, because the decision under test is a pure function of one
string. The wheel checks can only see it after a full wheel build, and only on the platform that
built one, so a filter that dropped the wrong entry reached the published artifact before anything
ran that would notice.

Two properties are covered, and they pull in opposite directions, which is why both are needed.
The filter must drop the MKL arch directories torch's exported link interface leaves anchored at
the filesystem root, and it must keep the absolute torch directory that is a library's only route
to torch when no relative one was recorded. A filter that satisfies either alone is wrong: the
first way ships unusable paths, the second way stops the extensions importing.

The functions are read out of setup.py rather than restated, so the test exercises what ships.
setup.py calls setup() at module scope, so it is loaded by compiling the definitions this needs
instead of importing it, which would exit during setuptools argument parsing.
"""

import ast
import importlib.util
import re
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Compiled from setup.py, so a change to the filter is exercised here rather than duplicated.
_WANTED = (
    "_MKL_ARCH_DIRECTORIES",
    "_is_cuda_toolkit_directory",
    "_is_unresolved_math_library_directory",
    "_is_usable_runtime_path",
)


def _setup_source() -> str:
    """setup.py's text, decoded as UTF-8.

    The encoding is named because `read_text()` defaults to the locale's, and both files this test
    parses contain non-ASCII characters. On a Windows runner that resolves to a code page, which
    mangles them, and the mangled text is what gets parsed.
    """
    return (REPO_ROOT / "setup.py").read_text(encoding="utf-8")


def _setup_namespace() -> dict:
    """The runtime path helpers from setup.py, compiled without running setup()."""
    tree = ast.parse(_setup_source())
    wanted = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _WANTED:
                wanted.append(node)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in _WANTED for name in names):
                wanted.append(node)
    found = set()
    for node in wanted:
        found.add(
            node.name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else next(t.id for t in node.targets if isinstance(t, ast.Name))
        )
    missing = sorted(set(_WANTED) - found)
    assert not missing, (
        f"setup.py no longer defines {missing} at module scope, so this test would silently "
        "check nothing. Update the names here to match."
    )
    namespace = {"re": re, "PurePosixPath": PurePosixPath}
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), "<setup.py>", "exec"),
        namespace,
    )
    return namespace


@pytest.fixture(scope="module")
def setup_helpers() -> dict:
    return _setup_namespace()


# setup.py's own arch names, read at import so the parametrized cases below are driven by the
# shipped constant rather than a second copy of it. Adding an arch to setup.py then extends both
# the reject and the accept cases, which is what keeps packaging and the release check in step.
_MKL_ARCH_DIRECTORIES = _setup_namespace()["_MKL_ARCH_DIRECTORIES"]

# What the linker records when MKL's prefix resolves empty, leaving its arch subdirectory
# concatenated onto nothing. Two of the three name a Windows layout, in a Linux wheel.
UNRESOLVED_MATH_DIRECTORIES = tuple(f"/lib/{arch}" for arch in _MKL_ARCH_DIRECTORIES)


@pytest.mark.parametrize("entry", UNRESOLVED_MATH_DIRECTORIES)
def test_drops_math_directories_with_an_empty_prefix(entry, setup_helpers):
    assert setup_helpers["_is_unresolved_math_library_directory"](entry) is True
    # A trailing separator is the same directory, which the path type normalises on its own. Asserted
    # because patchelf prints entries as recorded, so that spelling can genuinely arrive.
    assert setup_helpers["_is_unresolved_math_library_directory"](entry + "/") is True


@pytest.mark.parametrize(
    "entry",
    [
        # A real MKL installation spells the same arch directory below a prefix, and that
        # directory genuinely exists, so dropping it would break a library resolving through it.
        "/opt/intel/mkl/lib/intel64",
        "/opt/intel/oneapi/mkl/latest/lib/intel64",
        "/usr/lib/intel64",
        "/home/user/lib/intel64",
        # The arch name as a parent rather than as the entry itself.
        "/lib/intel64/extra",
        # Ordinary system directories, which differ from the bad entries only in the last part.
        "/lib",
        "/lib64",
        "/lib/x86_64-linux-gnu",
        # Relative entries are decided before this predicate is reached, but it must not claim
        # one, or a wheel's own hop into a directory named intel64 would be dropped.
        "$ORIGIN/../../lib",
        "$ORIGIN/lib/intel64",
    ],
)
def test_keeps_directories_that_name_a_real_prefix(entry, setup_helpers):
    assert setup_helpers["_is_unresolved_math_library_directory"](entry) is False
    # Also through the production predicate, not just the narrow one. Asserting only the narrow
    # predicate let _is_usable_runtime_path start rejecting an ordinary system directory with the
    # whole suite green, since nothing checked that these entries actually survive the filter.
    assert setup_helpers["_is_usable_runtime_path"](entry, True, True) is True


# Read off the pybindings extension in the published x86_64 CPU nightly, in the recorded order. The
# MKL block sits after five relative hops and before one. On this wheel the entries are simply dead,
# since nothing resolves through them, so dropping them costs a few wasted lookups at load time and
# removes paths a user cannot have. On the CUDA wheel the same block precedes the hop into the CUDA
# runtime, which is where the ordering matters.
SHIPPED_RUNTIME_PATH = [
    "$ORIGIN/../../../torch/lib",
    "$ORIGIN/../../src/executorch/lib",
    "$ORIGIN/../../backends/qualcomm",
    "$ORIGIN/../../lib",
    "$ORIGIN/../../../lib64",
    "/lib/intel64",
    "/lib/intel64_win",
    "/lib/win-x64",
    "$ORIGIN/../../backends/cuda",
]


def _relative_torch_route_predicate():
    """setup.py's own has_relative_torch_route expression, lifted out of its function.

    Read rather than restated, because it decides the third argument to the filter under test.
    A copy here would let the test keep passing after that expression changed, which is the one
    failure a regression guard must not have.
    """
    tree = ast.parse(_setup_source())
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "has_relative_torch_route"
                for target in node.targets
            )
        ):
            continue
        # The expression iterates a name bound in setup.py's own scope, so it is rebound to this
        # function's argument by compiling it as the body of a one-argument lambda.
        source = ast.unparse(node.value)
        iterated = "for entry in found"
        assert iterated in source, (
            f"setup.py computes has_relative_torch_route as {source!r}, which no longer iterates "
            "the name this test rebinds. Update the rebinding rather than leaving it a no-op."
        )
        return eval(
            f"lambda entries: {source.replace(iterated, 'for entry in entries')}"
        )
    raise AssertionError(
        "setup.py no longer computes has_relative_torch_route, so this test would pass the "
        "filter an argument the shipped code never produces."
    )


def _filtered(entries, setup_helpers):
    is_usable = setup_helpers["_is_usable_runtime_path"]
    has_relative_torch_route = _relative_torch_route_predicate()(entries)
    return [
        entry
        for entry in entries
        # True is safe_to_drop_toolkit_paths: this wheel ships no CUDA, so an absolute toolkit path
        # in it names only the build machine. Packaging derives the same value from the built tree.
        if is_usable(entry, True, has_relative_torch_route)
    ]


def test_shipped_library_keeps_no_absolute_entry(setup_helpers):
    kept = _filtered(SHIPPED_RUNTIME_PATH, setup_helpers)
    assert [entry for entry in kept if entry.startswith("/")] == []


def test_shipped_library_keeps_every_relative_hop(setup_helpers):
    # Asserted separately from the absence of absolute entries, because a filter that dropped
    # everything would satisfy that one while leaving the library unable to find its siblings.
    kept = _filtered(SHIPPED_RUNTIME_PATH, setup_helpers)
    assert kept == [
        entry for entry in SHIPPED_RUNTIME_PATH if not entry.startswith("/")
    ]


def test_keeps_the_absolute_torch_directory_when_it_is_the_only_route(setup_helpers):
    # The case that stops this being a blanket "drop everything absolute": an extension links
    # torch and, with no relative route recorded, reaches it only through the directory the
    # linker found it in. Dropping that would stop the extension importing.
    entries = [
        "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/lib",
        "/lib/intel64",
    ]
    assert _filtered(entries, setup_helpers) == [entries[0]]


def test_drops_the_absolute_torch_directory_when_a_relative_route_exists(setup_helpers):
    # The other side of the same rule, and the only case that exercises the route expression at all.
    # Without a relative entry in the list the route is False whatever that expression says, so the
    # test above cannot tell a correct expression from an inverted one.
    entries = [
        "$ORIGIN/../../../torch/lib",
        "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/lib",
        "/lib/intel64",
    ]
    assert _filtered(entries, setup_helpers) == [entries[0]]


_MATH_DIRECTORY_REASON = "a maths library directory whose prefix resolved empty"
_BUILD_DIRECTORY_REASON = "inside a build of this project"
_UNREACHABLE_REASON = "an absolute directory the wheel has a relative route to"


def _release_check_decision():
    """The release check's own per-entry decision, imported rather than replayed.

    Loaded as a module so the unit test exercises the function the wheel check calls, rather than
    that function's source text. Reading the text let the rejection be deleted outright.
    """
    path = REPO_ROOT / ".ci" / "scripts" / "wheel" / "test_shared_libraries.py"
    spec = importlib.util.spec_from_file_location("_release_check_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._unusable_runtime_path_kind


@pytest.mark.parametrize("arch", _MKL_ARCH_DIRECTORIES)
def test_release_check_rejects_the_math_directories(arch):
    # The check and packaging are separate code, so a wheel built without patchelf keeps these
    # entries and only the check would catch it. Driven off setup.py's own constant, and asserting
    # the REASON rather than merely a rejection: the check rejects every absolute path it does not
    # recognise, so an arch it was never taught about would otherwise pass through the catch-all.
    decide = _release_check_decision()
    assert (
        decide(f"/lib/{arch}", "_C.cpython-312-x86_64-linux-gnu.so")
        == _MATH_DIRECTORY_REASON
    )


def test_release_check_rejects_each_kind_for_its_own_reason():
    # One assertion per rejecting branch, by reason, so deleting any single branch fails here.
    # Asserting only "not None" let the build-directory branch and the catch-all each be removed
    # on their own with every test green.
    decide = _release_check_decision()
    assert decide("/home/u/pip-out/lib", "_C.so") == _BUILD_DIRECTORY_REASON
    assert decide("/opt/rocm/lib", "_C.so") == _UNREACHABLE_REASON
    assert decide("", "_C.so") is not None


def test_release_check_rejects_a_build_directory_before_the_allowlist():
    # Order is load bearing and nothing else holds it. A torch directory inside a CI worker tree
    # must be rejected, which only happens because the build-directory branch runs before the
    # suffix allowlist gets to accept the /torch/lib ending.
    decide = _release_check_decision()
    entry = "/home/ec2-user/actions-runner/_work/executorch/pytorch/torch/lib"
    assert decide(entry, "_C.so") == _BUILD_DIRECTORY_REASON


@pytest.mark.parametrize("arch", _MKL_ARCH_DIRECTORIES)
def test_release_check_still_accepts_a_real_mkl_installation(arch, setup_helpers):
    # The two must agree on every arch in the shared constant. Packaging KEEPS a prefixed one,
    # because the environment provides it, so a check that rejected it would fail a wheel packaging
    # deliberately allowed and the builder could not satisfy both. Parametrized off the constant, so
    # adding an arch to setup.py without teaching the check about it fails here.
    entry = f"/opt/intel/mkl/lib/{arch}"
    assert setup_helpers["_is_usable_runtime_path"](entry, True, True) is True
    assert (
        _release_check_decision()(entry, "_C.cpython-312-x86_64-linux-gnu.so") is None
    )


def test_the_wheel_scan_consults_the_classifier():
    # The unit tests above call the classifier directly, and the wheel scan is the only thing that
    # applies it to a real library. That scan needs an installed wheel, so it cannot run here;
    # what is checkable is the wiring, and severing it left all tests green. Asserted on the AST so
    # a rename or an accidental deletion fails rather than silently disabling the enforcement.
    path = REPO_ROOT / ".ci" / "scripts" / "wheel" / "test_shared_libraries.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scan = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "test_no_absolute_runtime_paths"
        ),
        None,
    )
    assert scan is not None, "the wheel scan this check enforces no longer exists"
    called = {
        node.func.id
        for node in ast.walk(scan)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_unusable_runtime_path_kind" in called, (
        "test_no_absolute_runtime_paths no longer calls _unusable_runtime_path_kind, so the wheel "
        "scan would report every shipped library clean while these unit tests still pass."
    )


def test_release_check_accepts_a_relative_entry():
    # A relative hop is the normal case and must never be rejected, whatever the absolute rules do.
    assert _release_check_decision()("$ORIGIN/../../lib", "_C.so") is None
