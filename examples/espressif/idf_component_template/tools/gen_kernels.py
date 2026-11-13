#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the kernel registration bundle for the ExecuTorch ESP-IDF component.

Runs ExecuTorch's selective-build codegen from the installed executorch package
over the given models and op lists, and writes the directory the component's
CMake consumes:

    python tools/gen_kernels.py --out main/et_kernels --pte main/model.pte
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

BUNDLE_SCHEMA = 1
LIBRARIES_SCHEMA = 1
REGISTRATION_FILES = (
    "RegisterCodegenUnboxedKernelsEverything.cpp",
    "NativeFunctions.h",
    "Functions.h",
)
MAX_KERNEL_HEADER = Path("include/executorch/runtime/kernel/selected_max_kernel_num.h")
# File browsers drop these into folders they open; they go with the old bundle.
OS_METADATA = {".DS_Store", "Thumbs.db", "desktop.ini"}


class GenKernelsError(Exception):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


def canonical_op_name(entry: dict) -> str:
    name = entry["op"] if "op" in entry else entry["func"].split("(", 1)[0]
    return name if "::" in name else f"aten::{name}"


def assign_ops(
    ops: Set[str], libraries: List[Tuple[str, Set[str]]], prim_ops: Set[str]
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """Splits ops into (ops per library, prims, unknown ops).

    Each op goes to the first library in `libraries` that declares it;
    _load_libraries() makes sure every library declaring it declares it alike.
    """
    by_library, prims, unknown = {}, [], []
    for op in sorted(ops):
        if op in prim_ops:
            prims.append(op)
            continue
        owner = next((name for name, declared in libraries if op in declared), None)
        if owner is None:
            unknown.append(op)
        else:
            by_library.setdefault(owner, []).append(op)
    ordered = {name: by_library[name] for name, _ in libraries if name in by_library}
    return ordered, prims, unknown


def versions_match(
    installed: str, installed_sha: str, component_version: str, component_sha: str
) -> bool:
    if installed_sha and installed_sha == component_sha:
        return True
    if "-dev." in component_version or "-rc." in component_version:
        return False
    # Release wheels report X.Y.Z (PyPI) or X.Y.Z+cpu (download.pytorch.org); a
    # source build reports X.Y.Z+<commit> and must match by commit.
    release = component_version.split("~", 1)[0]
    return installed in (release, f"{release}+cpu")


def _check_version(component_dir):
    import executorch.version

    try:
        manifest = yaml.safe_load((component_dir / "idf_component.yml").read_text())
        stamp = (component_dir / "VERSION").read_text()
        component_version = str(manifest["version"])
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        raise GenKernelsError(f"{component_dir} is not an ExecuTorch component: {e}")
    commit = re.search(r"^executorch_commit:\s*(\S+)", stamp, re.MULTILINE)
    component_sha = commit.group(1) if commit else ""
    installed = executorch.version.__version__
    installed_sha = executorch.version.git_version or ""
    if not versions_match(installed, installed_sha, component_version, component_sha):
        wanted = "the executorch the component was built from"
        if "-dev." in component_version:
            nightly = component_version.replace("-dev.", ".dev")
            wanted += (
                f" (for a published dev version, the nightly executorch=={nightly})"
            )
        raise GenKernelsError(
            f"installed executorch {installed} ({installed_sha[:12]}) does not match "
            f"the component's {component_version} ({component_sha[:12]}); export and "
            f"generate with {wanted}, installed as the component's README.md shows"
        )
    return component_version, component_sha, installed, installed_sha


def _load_libraries(component_dir):
    try:
        config = json.loads((component_dir / "tools/kernel_libraries.json").read_text())
    except (OSError, ValueError) as e:
        raise GenKernelsError(
            f"{component_dir} has no usable kernel_libraries.json: {e}"
        )
    if config.get("schema") != LIBRARIES_SCHEMA:
        raise GenKernelsError(
            f"unsupported kernel_libraries.json schema {config.get('schema')}"
        )
    libraries, declared = [], {}
    for library in config["libraries"]:
        yaml_path = component_dir / library["yaml"]
        entries = yaml.safe_load(yaml_path.read_text()) or []
        for entry in entries:
            op = canonical_op_name(entry)
            first, first_entry = declared.setdefault(op, (library["name"], entry))
            if first_entry != entry:
                raise GenKernelsError(
                    f"the {first} and {library['name']} kernel libraries declare "
                    f"{op} differently",
                    code=1,
                )
        libraries.append(
            (library["name"], yaml_path, {canonical_op_name(e) for e in entries})
        )
    prim_source = component_dir / config["prim_ops_source"]
    prim_ops = set(re.findall(r'\bKernel\(\s*"([^"]+)"', prim_source.read_text()))
    if not prim_ops:
        raise GenKernelsError(f"found no prim ops in {prim_source}", code=1)
    return libraries, prim_source, prim_ops


def _read_model(path):
    try:
        data = path.read_bytes()
    except OSError as e:
        raise GenKernelsError(f"cannot read {path}: {e.strerror}")
    if data[4:8] == b"BP08":
        raise GenKernelsError(
            f"{path} is a bundled program; pass the plain .pte exported with it"
        )
    from executorch.exir._serialize import _deserialize_pte_binary

    try:
        program = _deserialize_pte_binary(data).program
    except Exception as e:
        raise GenKernelsError(f"{path} is not an ExecuTorch .pte: {e}")
    backends = {d.id for plan in program.execution_plan for d in plan.delegates}
    return hashlib.sha256(data).hexdigest(), backends


def _check_out(out, inputs):
    """Refuses an --out that holds anything but a previous bundle."""
    for path in inputs:
        if out in path.parents:
            raise GenKernelsError(f"{path} is inside --out {out}; keep it elsewhere")
    if not out.exists():
        return
    if not out.is_dir():
        raise GenKernelsError(f"--out {out} is not a directory")
    if not any(out.iterdir()):
        return
    try:
        previous = json.loads((out / "kernels.json").read_text())
    except (OSError, ValueError):
        previous = None
    if not isinstance(previous, dict) or previous.get("schema") != BUNDLE_SCHEMA:
        raise GenKernelsError(
            f"{out} is not empty and is not a kernel bundle; refusing to overwrite it"
        )
    written = {Path("kernels.json"), Path("selected_prim_ops.h"), MAX_KERNEL_HEADER}
    written |= {
        Path(name, f)
        for name in previous.get("libraries", {})
        for f in REGISTRATION_FILES
    }
    written |= {parent for path in written for parent in path.parents}
    foreign = sorted(
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.relative_to(out) not in written and p.name not in OS_METADATA
    )
    if foreign:
        raise GenKernelsError(
            f"{out} holds files gen_kernels.py did not write ({', '.join(foreign)}); "
            "refusing to overwrite it"
        )


def _select(args, root_ops, tmp):
    """Writes one selection yaml per input; returns (op origins, models, backends, requested).

    `requested` is the ops that --ops and --oplist select, which only those
    inputs can reproduce.
    """
    from executorch.codegen.tools import gen_all_oplist, gen_oplist

    sources, models, backends = {}, set(), set()
    for pte in dict.fromkeys(p.resolve() for p in args.pte):
        sha, ids = _read_model(pte)
        models.add((pte.name, sha))
        backends |= ids
        selection = tmp / f"model{len(sources)}.yaml"
        gen_oplist.gen_oplist(selection, model_file_path=str(pte))
        sources[selection] = str(pte)
    if root_ops:
        selection = tmp / "ops.yaml"
        gen_oplist.gen_oplist(selection, root_ops=",".join(root_ops))
        sources[selection] = "--ops"
    for oplist in args.oplist:
        if not oplist.is_file():
            raise GenKernelsError(f"cannot read {oplist}")
        sources[oplist] = str(oplist)

    (tmp / "union").mkdir()
    gen_all_oplist.gen_all_oplist(str(tmp / "union"), [str(p) for p in sources])
    union = yaml.safe_load((tmp / "union/selected_operators.yaml").read_text())
    if union.get("include_all_operators"):
        raise GenKernelsError(
            "a selection includes all operators; list the operators instead"
        )
    origins = {}
    for selection, label in sources.items():
        for op in yaml.safe_load(selection.read_text()).get("operators") or {}:
            origins.setdefault(op, []).append(label)
    named = set(sources.values()) - {str(p.resolve()) for p in args.pte}
    requested = sorted(op for op, labels in origins.items() if named & set(labels))
    return origins, models, backends, requested


def _codegen(libraries, by_library, component_dir, tmp, staging):
    """Generates each selected library's registration; returns their oplists."""
    from executorch.codegen import gen
    from executorch.codegen.tools import gen_oplist

    oplists = []
    for name, yaml_path, _ in libraries:
        if name not in by_library:
            continue
        work = tmp / name
        (work / "gen").mkdir(parents=True)
        oplist = work / "selected_operators.yaml"
        gen_oplist.gen_oplist(oplist, root_ops=",".join(by_library[name]))
        gen.generate(
            source_path=str(component_dir / "tools/codegen"),
            install_dir=str(work / "gen"),
            functions_yaml_path=str(yaml_path),
            op_selection_yaml_path=str(oplist),
        )
        (staging / name).mkdir()
        for f in REGISTRATION_FILES:
            shutil.copyfile(work / "gen" / f, staging / name / f)
        oplists.append(oplist)
    return oplists


def generate(args):
    from executorch.codegen.tools import (
        gen_max_kernel_num,
        gen_oplist,
        gen_selected_prim_ops,
    )

    component_dir = args.component_dir.resolve()
    out = args.out.resolve()
    component_version, component_sha, installed, installed_sha = _check_version(
        component_dir
    )
    _check_out(out, [p.resolve() for p in args.pte + args.oplist])
    root_ops = sorted({op.strip() for arg in args.ops for op in arg.split(",")} - {""})
    if not (args.pte or root_ops or args.oplist):
        raise GenKernelsError("nothing to select: pass --pte, --ops or --oplist")
    libraries, prim_source, prim_ops = _load_libraries(component_dir)

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            origins, models, backends, requested = _select(args, root_ops, tmp)
            by_library, prims, unknown = assign_ops(
                set(origins), [(n, ops) for n, _, ops in libraries], prim_ops
            )
            if unknown:
                missing = ", ".join(
                    f"{op} (from {', '.join(origins[op])})" for op in unknown
                )
                raise GenKernelsError(f"no kernel library provides {missing}")

            oplists = _codegen(libraries, by_library, component_dir, tmp, staging)
            if not oplists:
                oplists.append(tmp / "none.yaml")
                gen_oplist.gen_oplist(oplists[0])
            gen_selected_prim_ops.write_selected_prim_ops(prims, str(staging))
            gen_max_kernel_num.gen_max_kernel_num(
                oplist_yamls=oplists,
                prim_ops_source=prim_source,
                output_path=staging / MAX_KERNEL_HEADER,
            )

        bundle = {
            "schema": BUNDLE_SCHEMA,
            "component_version": component_version,
            "component_commit": component_sha,
            "executorch_version": installed,
            "executorch_git_sha": installed_sha,
            "models": [{"name": n, "sha256": h} for n, h in sorted(models)],
            "libraries": by_library,
            "prim_ops": prims,
            "ops": requested,
            "backends": sorted(backends),
        }
        (staging / "kernels.json").write_text(json.dumps(bundle, indent=2) + "\n")
        umask = os.umask(0)
        os.umask(umask)
        staging.chmod(0o777 & ~umask)  # mkdtemp makes it 0700
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return bundle


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="bundle directory")
    parser.add_argument(
        "--pte", type=Path, action="append", default=[], help="model to cover"
    )
    parser.add_argument(
        "--ops",
        action="append",
        default=[],
        help="comma-separated operator names, e.g. aten::add.out",
    )
    parser.add_argument(
        "--oplist",
        type=Path,
        action="append",
        default=[],
        help="selected_operators.yaml from ExecuTorch's gen_oplist",
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the ExecuTorch component (default: the one holding this script)",
    )
    args = parser.parse_args(argv)
    try:
        bundle = generate(args)
    except GenKernelsError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    libraries = ", ".join(f"{n} ({len(o)} ops)" for n, o in bundle["libraries"].items())
    print(f"wrote {args.out}: {libraries or 'no kernel libraries'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
