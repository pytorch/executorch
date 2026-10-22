#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks the wheel is usable with only the dependencies it declares.

Why this exists, and why the other wheel checks cannot cover it: they run inside the build
checkout, which has every development requirement installed and the source tree on the path.
A module that imports a package the wheel never declares still passes there, and then fails
for a user who only ran `pip install executorch`. Two such gaps shipped, one of them in the
import path of XNNPACK, the default delegate.

The check asks a different question from "does this import here", which is always yes in a
build environment. It hides every distribution the wheel does not declare, then imports. So an
import that quietly relied on a development requirement fails here the way it fails for a
user, and names the package it wanted.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

# Modules a user reaches for after `pip install executorch`. Each must import with only the
# wheel's declared dependencies present.
#
# Limited to entry points that are documented and platform independent. A delegate whose
# availability depends on a separate SDK, an extra, or the host architecture is deliberately
# absent, because its absence is a legitimate state and asserting on it would fail for the
# wrong reason. What this list asserts is that nothing on it needs an undeclared package.
REQUIRED_IMPORTS: List[str] = [
    # The core export path.
    "executorch.exir",
    "executorch.export",
    # The runtime bindings, which are what load and run a .pte.
    "executorch.extension.pybindings.portable_lib",
    # XNNPACK is the default delegate and ships on every platform.
    "executorch.backends.xnnpack.partition.xnnpack_partitioner",
]

# Distributions that are legitimately present without being declared.
#
# torch, because the wheel deliberately does not declare it: a consumer brings the build
# matching their platform and accelerator. The rest are what torch itself requires, so they are
# guaranteed alongside it.
ASSUMED_PRESENT: Set[str] = {"torch", "executorch"}


def _normalise(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def _requirements_of(distribution: str) -> Set[str]:
    import importlib.metadata as metadata

    from packaging.requirements import Requirement

    try:
        raw_requirements = metadata.requires(distribution) or []
    except metadata.PackageNotFoundError:
        return set()

    names = set()
    for raw in raw_requirements:
        requirement = Requirement(raw)
        # An extra is opt in, so a module that needs one is not a clean install failure. Its
        # own absence is what this check is happy to see.
        if requirement.marker is not None and "extra" in str(requirement.marker):
            continue
        names.add(_normalise(requirement.name))
    return names


def _allowed_distributions() -> Set[str]:
    """Every distribution a clean install is entitled to import.

    The wheel's own declared requirements and torch, then the full transitive closure of both,
    because pip installs a dependency's dependencies too. Computed to a fixed point rather than
    a fixed depth: `six` arrives three levels down, through pandas and python-dateutil, and a
    depth limited walk reported it as undeclared.
    """
    allowed = {_normalise(name) for name in ASSUMED_PRESENT}
    allowed |= _requirements_of("executorch")
    allowed |= _requirements_of("torch")

    pending = set(allowed)
    while pending:
        allowed |= pending
        discovered = set()
        for name in pending:
            discovered |= _requirements_of(name)
        pending = discovered - allowed
    return allowed


def _import_names(distribution: str) -> Set[str]:
    """The import names a distribution provides.

    Read from the installed file list rather than top_level.txt, because most distributions
    built by a modern backend no longer write that file, and the ones that matter here are
    exactly the ones whose import name differs from their project name. `protobuf` provides
    `google`, `scikit-learn` provides `sklearn`, `attrs` provides `attr`, and `torch` also
    provides `functorch` and `torchgen`. Guessing from the project name would leave all of
    those unrecognised, and a name left unrecognised on the permitted side gets hidden, which
    fails a correct wheel.
    """
    import importlib.metadata as metadata

    try:
        files = metadata.distribution(distribution).files or []
    except metadata.PackageNotFoundError:
        return set()

    names = set()
    for entry in files:
        path = str(entry)
        # Outside the site directory, or not importable.
        if path.startswith("..") or path.startswith("__pycache__"):
            continue
        head = path.split("/")[0]
        if head.endswith((".dist-info", ".egg-info", ".data")):
            continue
        if head.endswith(".py"):
            names.add(head[: -len(".py")])
        elif head.endswith((".so", ".dylib", ".pyd")):
            # An extension module carries a platform tag, as in _foo.cpython-312-darwin.so.
            names.add(head.split(".")[0])
        elif "." not in head:
            names.add(head)
    return names


def _blocked_modules(allowed: Set[str]) -> Set[str]:
    """Import names in this environment that a clean install would not have.

    Built by elimination rather than by listing what to hide, then narrowed, because a wrong
    entry here fails a correct wheel rather than catching anything.
    """
    import importlib.metadata as metadata

    allowed_modules = set()
    for name in allowed:
        allowed_modules |= _import_names(name)
        allowed_modules.add(name)

    blocked = set()
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if not name or _normalise(name) in allowed:
            continue
        blocked |= _import_names(name) or {_normalise(name)}

    # Never hide a name a permitted distribution also provides. Two distributions can share a
    # namespace package, `google` being the common one, and blocking the shared name would
    # break the permitted one.
    blocked -= allowed_modules

    # Never hide anything the standard library provides, whatever a distribution claims. A
    # backport publishes the same import name as the module it backports, so blocking it would
    # remove a name that is present on a clean install anyway.
    blocked -= set(sys.stdlib_module_names)

    return blocked


def run_tests(work_dir: Path) -> None:
    allowed = _allowed_distributions()
    blocked = _blocked_modules(allowed)
    print(
        f"A clean install may import {len(allowed)} distributions; hiding "
        f"{len(blocked)} module names this environment adds"
    )

    # Somewhere with no checkout, so `import executorch` resolves to the installed package.
    # The probe drops that directory from sys.path itself, so no stray file can shadow a real
    # module either. Done in the probe rather than with the interpreter's -P flag, which only
    # exists from Python 3.11 and the wheel supports 3.10.
    neutral_dir = work_dir / "neutral"
    neutral_dir.mkdir(parents=True, exist_ok=True)

    failures: Dict[str, str] = {}
    for module in REQUIRED_IMPORTS:
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE],
            input=json.dumps({"module": module, "blocked": sorted(blocked)}),
            cwd=os.fspath(neutral_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures[module] = (result.stderr or result.stdout).strip()

    if failures:
        report = "\n\n".join(
            f"{name} failed to import:\n{err}" for name, err in failures.items()
        )
        raise AssertionError(
            f"{len(failures)} of {len(REQUIRED_IMPORTS)} modules are not usable from a clean "
            f"install, so a user who runs `pip install executorch` cannot use them. Either "
            f"declare the package they import, or import it lazily so the module itself still "
            f"loads.\n\n{report}"
        )

    _check_top_level_names()
    print(
        f"All {len(REQUIRED_IMPORTS)} modules import with only declared dependencies, and the "
        f"metadata names only the package."
    )


# Reads its arguments from stdin, blocks the named modules, then imports. A blocked module
# raises ModuleNotFoundError exactly as it would be absent, so the traceback shows the import
# chain that wanted it.
_IMPORT_PROBE = """
import importlib.abc, importlib.util, json, os, sys

# Drop the working directory, which `python -c` puts first. The equivalent flag, -P, is only
# available from 3.11.
for entry in ('', '.', os.getcwd()):
    while entry in sys.path:
        sys.path.remove(entry)

request = json.load(sys.stdin)
blocked = set(request["blocked"])


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(
                f"No module named {fullname!r} (not declared by the executorch wheel)",
                name=fullname,
            )
        return None


sys.meta_path.insert(0, _Blocker())
importlib.import_module(request["module"])
"""


def _check_top_level_names() -> None:
    """The distribution must claim exactly one top level name.

    Read from the installed metadata rather than the wheel file, because that is what any tool
    consuming it sees. This once listed twenty build recipe strings beside the real name.
    """
    import importlib.metadata as metadata

    text = metadata.distribution("executorch").read_text("top_level.txt") or ""
    names = [line for line in text.splitlines() if line.strip()]
    assert names == ["executorch"], (
        f"the installed wheel claims the top level names {names}, but only 'executorch' is "
        f"importable, so anything reading this metadata is misled"
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run_tests(Path(tmp))
    sys.exit(0)
