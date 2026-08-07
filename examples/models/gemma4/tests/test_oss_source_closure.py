# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""D10 OSS source closure and plain-Gemma non-disturbance contract."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import unittest

from pathlib import Path
from types import ModuleType

_XPLAT_PREFIX = ("xplat", "executorch")
_FBCODE_PREFIX = ("fbcode", "executorch")

_SL_STATUS_REASON = "list D10 changed paths for the OSS closure gate - sl help status"
_SL_LOG_REASON = "find the commit that introduced this gate - sl help log"

_D10_SURFACE_DIRECTORIES = (
    "backends/webgpu/scripts",
    "backends/webgpu/test",
    "examples/models/gemma4",
)

_D10_SURFACE_FILES = ("backends/webgpu/CMakeLists.txt",)

_THIS_GATE = "examples/models/gemma4/tests/test_oss_source_closure.py"

# D10's changed files, executorch-relative. The drift guard below compares this
# list against source control wherever source control can still identify D10.
_D10_CHANGED_PATHS = (
    "backends/webgpu/CMakeLists.txt",
    "backends/webgpu/scripts/test_gemma4_wasm_factory_contract.sh",
    "backends/webgpu/scripts/test_webgpu_native_ci.sh",
    "backends/webgpu/test/BUCK",
    "backends/webgpu/test/native/test_q4gsw_m3.cpp",
    "backends/webgpu/test/native/test_scatter.cpp",
    "backends/webgpu/test/native/test_topk.cpp",
    "backends/webgpu/test/op_tests/cases.py",
    "backends/webgpu/test/op_tests/test_typed_input_contract.py",
    "backends/webgpu/test/ops/index/test_index.py",
    "backends/webgpu/test/ops/scatter/__init__.py",
    "backends/webgpu/test/ops/scatter/export_scatter_artifacts.py",
    "backends/webgpu/test/ops/scatter/test_scatter.py",
    "backends/webgpu/test/ops/test_gather.py",
    "backends/webgpu/test/ops/test_to_copy.py",
    "backends/webgpu/test/ops/test_where.py",
    "backends/webgpu/test/ops/topk/__init__.py",
    "backends/webgpu/test/ops/topk/export_topk_artifacts.py",
    "backends/webgpu/test/ops/topk/test_topk.py",
    "backends/webgpu/test/targets.bzl",
    "backends/webgpu/test/test_native_ci_contract.py",
    "examples/models/gemma4/tests/generate_mtp_spec_oracle.py",
    "examples/models/gemma4/tests/targets.bzl",
    "examples/models/gemma4/tests/test_eagle_combined_round.py",
    "examples/models/gemma4/tests/test_export_assistant_webgpu_artifacts.py",
    "examples/models/gemma4/tests/test_export_partitioners.py",
    "examples/models/gemma4/tests/test_gemma4_spec_runner_contract.cpp",
    "examples/models/gemma4/tests/test_mtp_spec_oracle.py",
    "examples/models/gemma4/tests/test_oss_source_closure.py",
    "examples/models/gemma4/tests/test_webgpu_artifact_manifest.py",
    "examples/models/gemma4/tests/test_webgpu_spec_contract.py",
)

_D10_COMMAND_MODULES = (
    "executorch.backends.webgpu.test.ops.scatter.export_scatter_artifacts",
    "executorch.backends.webgpu.test.ops.topk.export_topk_artifacts",
    "executorch.backends.webgpu.test.ops.topk.test_topk",
    "executorch.backends.webgpu.test.op_tests.generate_op_tests",
)

_D10_COMMAND_SCRIPTS = ("examples/models/gemma4/tests/generate_mtp_spec_oracle.py",)

_D10_COMMAND_TEST_CASE = (
    "executorch.backends.webgpu.test.ops.topk.test_topk",
    "TestEagleTopKCpu",
    "test_eager_reference_is_repeatable",
)

# Each pattern splits one character so this file is not its own violation.
_INTERNAL_REFERENCE_PATTERNS = (
    r"manifol[d]",
    r"internalf[b]",
    r"fbur[l]",
    r"/data/user[s]/",
    r"/hom[e]/",
    r"/User[s]/",
    r"/mn[t]/",
    r"examples/models/f[b]/",
    r"\bD1[0-9]{8}\b",
    r"localhos[t]:[0-9]+",
    r"127[.]0[.]0[.]1:[0-9]+",
    r"\.intern[.]facebook[.]com",
)

# Rejection tests must name the strings they reject; only these two may do so.
_NEGATIVE_FIXTURE_MARKER = "oss-closure-" "fixture"  # split: not self-marking
_NEGATIVE_FIXTURE_FILES = (
    "examples/models/gemma4/tests/test_export_assistant_webgpu_artifacts.py",
    "examples/models/gemma4/tests/test_webgpu_artifact_manifest.py",
)

_BINARY_ARTIFACT_SUFFIXES = (
    ".pte",
    ".ptd",
    ".bin",
    ".gguf",
    ".safetensors",
    ".png",
    ".wasm",
)

_PLAIN_PARTITIONER = "examples/models/gemma4/webgpu_partitioner.py"
_PLAIN_MANIFEST_MODULE = "examples/models/gemma4/webgpu_artifact_manifest.py"
_PLAIN_MANIFEST_JSON = "examples/models/gemma4/manifests/gemma4_e2b_webgpu.json"
_PLAIN_RUNNER_HEADER = "examples/models/gemma4/runner/gemma4_runner.h"
_PLAIN_RUNNER_SOURCE = "examples/models/gemma4/runner/gemma4_runner.cpp"
_PLAIN_MODEL_TARGETS = "examples/models/gemma4/targets.bzl"

_PLAIN_WEBGPU_ALLOWLIST = (
    "exir_ops.edge.aten._assert_scalar.default",
    "exir_ops.edge.aten.add.Tensor",
    "exir_ops.edge.aten.argmax.default",
    "exir_ops.edge.aten.cat.default",
    "exir_ops.edge.aten.clamp.default",
    "exir_ops.edge.aten.div.Tensor",
    "exir_ops.edge.aten.gelu.default",
    "exir_ops.edge.aten.mul.Tensor",
    "exir_ops.edge.aten.permute_copy.default",
    "exir_ops.edge.aten.select_copy.int",
    "exir_ops.edge.aten.sigmoid.default",
    "exir_ops.edge.aten.slice_copy.Tensor",
    "exir_ops.edge.aten.squeeze_copy.dims",
    "exir_ops.edge.aten.sym_constrain_range_for_size.default",
    "exir_ops.edge.aten.tanh.default",
    "exir_ops.edge.aten.unsqueeze_copy.default",
    "exir_ops.edge.aten.view_copy.default",
    "exir_ops.edge.dim_order_ops._clone_dim_order.default",
    "exir_ops.edge.dim_order_ops._to_dim_order_copy.default",
    "exir_ops.edge.et_vk.apply_rotary_emb_hf.default",
    "exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default",
    "exir_ops.edge.et_vk.gemma4_sdpa.default",
    "exir_ops.edge.et_vk.rms_norm.default",
    "exir_ops.edge.et_vk.select_as_symint.default",
)

_PLAIN_EXPORT_CONTRACT_SHA256 = (
    "7b1e7ad9753bdc937f32a254eff04dad10528ac3c704f0f176894c2c435c9f2d"
)
_PLAIN_ARCHITECTURE_SHA256 = (
    "d731d17637aca0e808a6bdca6b80231310b84b6ed1708c579d95ff96c470d1a8"
)
_PLAIN_ACQUISITION_SHA256 = (
    "f1aad2baf1b48edf5124b993208f73ac2ae6878d890a84ab0e211d05babf316a"
)


def _tree_root() -> tuple[str, Path]:
    parents = list(Path(__file__).resolve().parents)
    for parent in parents:
        if (parent.joinpath(*_XPLAT_PREFIX).is_dir()) and (
            parent.joinpath(*_FBCODE_PREFIX).is_dir()
        ):
            return "fbsource", parent
    for parent in parents:
        if (parent / "backends" / "webgpu").is_dir() and (
            parent / "examples" / "models" / "gemma4"
        ).is_dir():
            return "oss", parent
    raise RuntimeError(f"no fbsource or OSS root above {Path(__file__).resolve()}")


def _physical_paths(layout: str, root: Path, relative: str) -> dict[str, Path]:
    if layout == "oss":
        return {"oss": root / relative}
    return {
        "xplat": root.joinpath(*_XPLAT_PREFIX) / relative,
        "fbcode": root.joinpath(*_FBCODE_PREFIX) / relative,
    }


def _canonical_paths(layout: str, root: Path, relative: str) -> Path:
    forms = _physical_paths(layout, root, relative)
    return forms["oss"] if layout == "oss" else forms["xplat"]


def _identity_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _dotted_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise ValueError(f"not an attribute chain: {ast.dump(node)}")
    parts.append(node.id)
    return ".".join(reversed(parts))


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_constant(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"module constant not found: {name}")


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _method_def(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == name:
                    return member
    raise AssertionError(f"method not found: {class_name}.{name}")


def _returned_dotted_names(function: ast.FunctionDef) -> tuple[str, ...]:
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.List):
            return tuple(_dotted_name(element) for element in node.value.elts)
    raise AssertionError(f"{function.name} does not return a list literal")


def _sl_lines(root: Path, args: tuple[str, ...], reason: str) -> tuple[str, ...] | None:
    """Stdout lines of one read-only `sl` call; None when `sl` cannot answer."""
    try:
        completed = subprocess.run(
            ["sl", *args, "--reason", reason],
            capture_output=True,
            cwd=root,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return tuple(completed.stdout.splitlines())


def _status_paths(
    root: Path, codes: tuple[str, ...] = ("M ", "A ")
) -> frozenset[str] | None:
    """Repo-relative paths `sl status` reports under the given status codes."""
    lines = _sl_lines(root, ("status",), _SL_STATUS_REASON)
    if lines is None:
        return None
    return frozenset(line[2:] for line in lines if line[:2] in codes)


def _introducing_commit(root: Path) -> str:
    """Node that last touched this gate; empty while D10 is uncommitted."""
    relative = str(Path(__file__).resolve().relative_to(root))
    lines = _sl_lines(
        root, ("log", relative, "-T", "{node}\n", "-l", "1"), _SL_LOG_REASON
    )
    return lines[0] if lines else ""


def _sapling_is_usable(root: Path) -> bool:
    """True when `sl` and a Sapling working copy are both present at root."""
    if shutil.which("sl") is None:
        return False
    return any(
        (parent / ".sl").is_dir() or (parent / ".hg").is_dir()
        for parent in (root, *root.parents)
    )


def _executorch_relative(layout: str, repo_path: str) -> str | None:
    if layout == "oss":
        return repo_path
    for parts in (_XPLAT_PREFIX, _FBCODE_PREFIX):
        prefix = "/".join(parts) + "/"
        if repo_path.startswith(prefix):
            return repo_path[len(prefix) :]
    return None


def _in_d10_surface(relative: str) -> bool:
    if relative in _D10_SURFACE_FILES:
        return True
    return any(
        relative.startswith(directory + "/") for directory in _D10_SURFACE_DIRECTORIES
    )


def _untracked_relatives(layout: str, root: Path) -> frozenset[str]:
    """Executorch-relative paths source control reports as not tracked."""
    reported = _status_paths(root, ("? ",))
    if reported is None:
        return frozenset()
    relatives: set[str] = set()
    for repo_path in reported:
        relative = _executorch_relative(layout, repo_path)
        if relative is not None:
            relatives.add(relative)
    return frozenset(relatives)


def _surface_status_paths(layout: str, root: Path) -> frozenset[str] | None:
    """Reported changes narrowed to D10's surface; None when `sl` cannot answer."""
    reported = _status_paths(root)
    if reported is None:
        return None
    relatives: set[str] = set()
    for repo_path in reported:
        relative = _executorch_relative(layout, repo_path)
        if relative is not None and _in_d10_surface(relative):
            relatives.add(relative)
    return frozenset(relatives)


def _install_executorch_namespace(layout: str, root: Path) -> None:
    """Bind `executorch.*` to this tree so a by-path load resolves its imports."""
    if "executorch" in sys.modules:
        return
    base = root / "xplat" / "executorch" if layout == "fbsource" else root
    package = ModuleType("executorch")
    package.__path__ = [str(base)]  # pyre-ignore[16]
    sys.modules["executorch"] = package


def _load_plain_manifest_module(layout: str, root: Path) -> ModuleType:
    """Load by path; the gemma4 package __init__ pulls in torch."""
    _install_executorch_namespace(layout, root)
    path = _canonical_paths(layout, root, _PLAIN_MANIFEST_MODULE)
    spec = importlib.util.spec_from_file_location(
        "gemma4_plain_webgpu_artifact_manifest", path
    )
    if spec is None:
        raise AssertionError(f"cannot load plain manifest module: {path}")
    loader = spec.loader
    if loader is None:
        raise AssertionError(f"cannot load plain manifest module: {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class OssSourceClosureTest(unittest.TestCase):
    maxDiff: int | None = None

    def setUp(self) -> None:
        self.layout, self.root = _tree_root()
        self.changed = _D10_CHANGED_PATHS

    def _existing_copies(self) -> list[tuple[str, str, Path]]:
        copies: list[tuple[str, str, Path]] = []
        for relative in self.changed:
            for form, path in _physical_paths(self.layout, self.root, relative).items():
                if path.is_file():
                    copies.append((relative, form, path))
        return copies

    def test_every_d10_path_has_its_required_physical_form(self) -> None:
        missing = [
            relative
            for relative in self.changed
            if not _canonical_paths(self.layout, self.root, relative).is_file()
        ]
        self.assertEqual(missing, [], f"{self.layout} canonical copies missing")

    def test_every_committed_d10_path_is_inside_the_d10_surface(self) -> None:
        outside = [r for r in self.changed if not _in_d10_surface(r)]
        self.assertEqual(
            outside,
            [],
            "the committed D10 path list left the surface this gate scans; the "
            "drift guard narrows source control to that surface, so a path "
            "outside it would never be compared",
        )

    def test_committed_d10_paths_match_the_uncommitted_working_copy(self) -> None:
        """Strict where source control can still identify D10; else says so."""
        reported = _surface_status_paths(self.layout, self.root)
        if reported is None:
            self.assertFalse(
                _sapling_is_usable(self.root),
                "`sl status` failed inside a Sapling working copy, so the "
                "committed D10 path list went unverified here",
            )
            return
        if _THIS_GATE not in reported:
            self.assertNotEqual(
                _introducing_commit(self.root),
                "",
                "this gate is neither an uncommitted change nor a committed "
                "file, so the committed D10 path list went unverified here",
            )
            return
        self.assertEqual(
            sorted(reported),
            list(self.changed),
            "the committed D10 path list drifted from the working copy",
        )

    def test_fbcode_mirror_is_byte_identical_where_present(self) -> None:
        if self.layout != "fbsource":
            self.assertEqual(self.layout, "oss")
            return
        # Sapling mirrors xplat->fbcode on commit, so a path that is still an
        # uncommitted working-copy change legitimately has no matching mirror.
        uncommitted = _status_paths(self.root)
        if uncommitted is None:
            self.assertFalse(
                _sapling_is_usable(self.root),
                "`sl status` failed inside a Sapling working copy, so the paths "
                "the mirror legitimately lags could not be identified",
            )
            return
        differing: list[str] = []
        absent_precommit: list[str] = []
        for relative in self.changed:
            forms = _physical_paths(self.layout, self.root, relative)
            xplat, fbcode = forms["xplat"], forms["fbcode"]
            if not xplat.is_file():
                continue
            if str(xplat.relative_to(self.root)) in uncommitted:
                continue
            if not fbcode.is_file():
                absent_precommit.append(relative)
                continue
            if xplat.read_bytes() != fbcode.read_bytes():
                differing.append(relative)
        self.assertEqual(
            differing,
            [],
            "xplat/fbcode mirror diverges; still-uncommitted paths are exempt "
            f"and the mirror is absent for: {absent_precommit}",
        )

    def test_no_internal_only_reference_in_d10_sources(self) -> None:
        violations: list[str] = []
        for relative, form, path in self._existing_copies():
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if _NEGATIVE_FIXTURE_MARKER in line:
                    continue
                for pattern in _INTERNAL_REFERENCE_PATTERNS:
                    found = re.search(pattern, line, re.IGNORECASE)
                    if found is not None:
                        violations.append(
                            f"{form}:{relative}:{number}: {found.group(0)}"
                        )
                        break
        self.assertEqual(violations, [], "internal-only reference")

    def test_negative_fixture_marker_cannot_silence_other_files(self) -> None:
        marked = {
            relative
            for relative, _form, path in self._existing_copies()
            if _NEGATIVE_FIXTURE_MARKER
            in path.read_text(encoding="utf-8", errors="replace")
        }
        self.assertEqual(marked, set(_NEGATIVE_FIXTURE_FILES))

    def test_no_binary_artifact_in_d10_paths_or_directories(self) -> None:
        untracked = _untracked_relatives(self.layout, self.root)
        binaries = [
            relative
            for relative in self.changed
            if Path(relative).suffix in _BINARY_ARTIFACT_SUFFIXES
        ]
        for directory in _D10_SURFACE_DIRECTORIES:
            for form, base in _physical_paths(
                self.layout, self.root, directory
            ).items():
                if not base.is_dir():
                    continue
                for path in sorted(base.rglob("*")):
                    if "__pycache__" in path.parts or not path.is_file():
                        continue
                    if path.suffix not in _BINARY_ARTIFACT_SUFFIXES:
                        continue
                    if f"{directory}/{path.relative_to(base)}" in untracked:
                        continue
                    binaries.append(f"{form}:{path.relative_to(base)}")
        self.assertEqual(
            binaries, [], "committed binary artifact; untracked scratch is exempt"
        )

    def test_command_contract_module_paths_resolve(self) -> None:
        unresolved: list[str] = []
        for module in _D10_COMMAND_MODULES:
            self.assertTrue(module.startswith("executorch."), module)
            relative = module[len("executorch.") :].replace(".", "/") + ".py"
            if not _canonical_paths(self.layout, self.root, relative).is_file():
                unresolved.append(f"{module} -> {relative}")
        for relative in _D10_COMMAND_SCRIPTS:
            if not _canonical_paths(self.layout, self.root, relative).is_file():
                unresolved.append(relative)
        self.assertEqual(unresolved, [], "command contract path does not resolve")

    def test_command_contract_unittest_target_resolves(self) -> None:
        module, class_name, method = _D10_COMMAND_TEST_CASE
        relative = module[len("executorch.") :].replace(".", "/") + ".py"
        path = _canonical_paths(self.layout, self.root, relative)
        self.assertTrue(path.is_file(), relative)
        self.assertIsNotNone(_method_def(_module_tree(path), class_name, method))


class PlainGemma4ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout, self.root = _tree_root()
        self.changed = _D10_CHANGED_PATHS
        self.partitioner = _module_tree(
            _canonical_paths(self.layout, self.root, _PLAIN_PARTITIONER)
        )

    def test_plain_partitioner_expected_counts_are_unchanged(self) -> None:
        self.assertEqual(
            _module_constant(self.partitioner, "_EXPECTED_GEMMA4_SDPA_COUNT"), 35
        )
        self.assertEqual(
            _module_constant(self.partitioner, "_EXPECTED_SINGLE_HF_ROPE_COUNT"), 20
        )

    def test_plain_webgpu_allowlist_membership_is_unchanged(self) -> None:
        allowlist = _returned_dotted_names(
            _function_def(self.partitioner, "_webgpu_allowlist")
        )
        self.assertEqual(len(allowlist), len(_PLAIN_WEBGPU_ALLOWLIST))
        self.assertEqual(tuple(sorted(allowlist)), _PLAIN_WEBGPU_ALLOWLIST)

    def test_plain_webgpu_allowlist_has_no_mtp_operator(self) -> None:
        allowlist = _returned_dotted_names(
            _function_def(self.partitioner, "_webgpu_allowlist")
        )
        forbidden = [
            entry
            for entry in allowlist
            if "scatter" in entry or "topk" in entry.lower()
        ]
        self.assertEqual(forbidden, [], "MTP operator leaked into plain allowlist")

    def test_plain_partitioner_fails_closed_on_emb8(self) -> None:
        init = _method_def(self.partitioner, "Gemma4WebGPUPartitioner", "__init__")
        guards = [
            statement
            for statement in init.body
            if isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Compare)
            and isinstance(statement.test.left, ast.Constant)
            and statement.test.left.value == "emb8"
        ]
        self.assertEqual(len(guards), 1)
        guard = guards[0]
        test = guard.test
        self.assertIsInstance(test, ast.Compare)
        assert isinstance(test, ast.Compare)
        self.assertIsInstance(test.left, ast.Constant)
        assert isinstance(test.left, ast.Constant)
        self.assertEqual(test.left.value, "emb8")
        self.assertEqual(len(test.ops), 1)
        self.assertIsInstance(test.ops[0], ast.In)
        self.assertEqual(_dotted_name(test.comparators[0]), "text_quantize")
        raised = guard.body[0]
        self.assertIsInstance(raised, ast.Raise)
        assert isinstance(raised, ast.Raise)
        self.assertIsInstance(raised.exc, ast.Call)
        assert isinstance(raised.exc, ast.Call)
        self.assertEqual(_dotted_name(raised.exc.func), "ValueError")

    def test_plain_manifest_identities_are_unchanged(self) -> None:
        module = _load_plain_manifest_module(self.layout, self.root)
        self.assertEqual(module.WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES, 1_500_000_000)
        self.assertEqual(module.EXPORT_CONTRACT["methods"], ["text_decoder"])
        self.assertEqual(module.EXPORT_CONTRACT["max_input_len"], 512)
        self.assertEqual(module.EXPORT_CONTRACT["max_seq_len"], 8960)
        self.assertEqual(module.ARCHITECTURE_FINGERPRINT["num_hidden_layers"], 35)
        self.assertEqual(len(module.ARCHITECTURE_FINGERPRINT["layer_types"]), 35)
        self.assertEqual(
            module.CHECKPOINT_ACQUISITION["repo_id"],
            "google/gemma-4-E2B-it-qat-q4_0-unquantized",
        )
        self.assertEqual(
            _identity_sha256(module.EXPORT_CONTRACT), _PLAIN_EXPORT_CONTRACT_SHA256
        )
        self.assertEqual(
            _identity_sha256(module.ARCHITECTURE_FINGERPRINT),
            _PLAIN_ARCHITECTURE_SHA256,
        )
        self.assertEqual(
            _identity_sha256(module.CHECKPOINT_ACQUISITION),
            _PLAIN_ACQUISITION_SHA256,
        )

    def test_plain_manifest_requires_exactly_three_ordered_ptds(self) -> None:
        module = _load_plain_manifest_module(self.layout, self.root)
        roles = {"pte": Path("model.pte"), "source": Path("source.json")}
        for count in (0, 2, 4):
            with self.assertRaises(ValueError) as raised:
                module.create_plain_manifest(
                    self.root, roles, [Path(f"c{i}.ptd") for i in range(count)]
                )
            self.assertIn("exactly three ordered PTDs", str(raised.exception))

    def test_plain_manifest_json_matches_the_plain_validator(self) -> None:
        module = _load_plain_manifest_module(self.layout, self.root)
        path = _canonical_paths(self.layout, self.root, _PLAIN_MANIFEST_JSON)
        manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["export"]["methods"], ["text_decoder"])
        self.assertEqual(manifest["export"], module.EXPORT_CONTRACT)
        self.assertEqual(manifest["acquisition"], module.CHECKPOINT_ACQUISITION)
        self.assertEqual(
            manifest["model"]["architecture"], module.ARCHITECTURE_FINGERPRINT
        )
        self.assertEqual(
            manifest["model"]["source_config"],
            {
                "path": "config/e2b_config.json",
                "sha256": module.SOURCE_CONFIG_SHA256,
            },
        )
        self.assertEqual(manifest["schema_version"], 1)

        artifacts = manifest["artifacts"]
        ptds = [item for item in artifacts if item["role"] == "ptd"]
        self.assertEqual(len(manifest["ptd_order"]), 3)
        self.assertEqual([item["path"] for item in ptds], manifest["ptd_order"])
        for item in ptds:
            self.assertLess(
                item["bytes"], module.WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
            )
        self.assertIn("pte", {item["role"] for item in artifacts})
        for item in artifacts:
            self.assertEqual(len(Path(item["path"]).parts), 1, item["path"])

    def test_public_xnnpack_runner_is_untouched_by_d10(self) -> None:
        owned = [
            relative
            for relative in self.changed
            if relative.startswith("examples/models/gemma4/runner/")
        ]
        self.assertEqual(owned, [], "D10 must not own the public runner")

        header = _canonical_paths(
            self.layout, self.root, _PLAIN_RUNNER_HEADER
        ).read_text(encoding="utf-8")
        self.assertIn("bool enable_workspace_sharing = true", header)

        source = _canonical_paths(
            self.layout, self.root, _PLAIN_RUNNER_SOURCE
        ).read_text(encoding="utf-8")
        self.assertIn(
            "#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>",
            source,
        )
        self.assertIn(
            "executorch::backends::xnnpack::WorkspaceSharingMode::PerModel",
            source,
        )

        targets = _canonical_paths(
            self.layout, self.root, _PLAIN_MODEL_TARGETS
        ).read_text(encoding="utf-8")
        self.assertIn('"//executorch/backends/xnnpack:xnnpack_backend"', targets)
        self.assertIn('"//executorch/backends/xnnpack:xnnpack_interface"', targets)
