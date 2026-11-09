# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import sys

from pathlib import Path

import torch


ARM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ARM_ROOT.parents[1]
CHECKER_PATH = ARM_ROOT / "scripts" / "check_model_support.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_test_arm_check_model_support",
        CHECKER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_read_supported_public_apis_handles_multiple_aliases(tmp_path: Path) -> None:
    table = tmp_path / "support.md"
    table.write_text(
        "# test\n\n"
        "| PyTorch API | Support profile | DType | Quantization mode |\n"
        "| --- | --- | --- | --- |\n"
        "| `torch.add` / `+` | INT | `INT8` | 8x8 |\n"
        "| `torch.bitwise_or` / `\\|` | INT | `INT8` | 8x8 |\n",
        encoding="utf-8",
    )

    assert checker._read_supported_public_apis(table) == {
        "torch.add",
        "torch.bitwise_or",
    }


def test_collect_exported_operators_finds_sort() -> None:
    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            values, _ = torch.sort(torch.relu(x + 1.0), dim=-1)
            return values

    exported_ops = checker._collect_exported_operators(
        Model(),
        (torch.randn(1, 8),),
    )

    assert "torch.ops.aten.relu.default" in exported_ops
    assert "torch.ops.aten.sort.default" in exported_ops


def test_find_unmatched_export_operators_reports_missing_api() -> None:
    aliases = {
        "torch.ops.aten.add.Tensor": ("torch.add", "+"),
        "torch.ops.aten.sort.default": ("torch.sort",),
    }

    unmatched = checker._find_unmatched_export_operators(
        aliases,
        {"torch.add"},
        lambda op: aliases[op],
    )

    assert unmatched == [
        checker.UnmatchedExportOperator(
            exported_op="torch.ops.aten.sort.default",
            public_apis=("torch.sort",),
        )
    ]


def test_assert_tensor_metadata_is_ignored_before_alias_resolution() -> None:
    metadata_op = "torch.ops.aten._assert_tensor_metadata.default"

    def unexpected_resolver(_op: str) -> tuple[str, ...]:
        raise AssertionError("ignored metadata op must not require alias resolution")

    assert (
        checker._find_unmatched_export_operators(
            [metadata_op],
            set(),
            unexpected_resolver,
        )
        == []
    )


def test_ignored_metadata_op_does_not_hide_real_unmatched_op() -> None:
    metadata_op = "torch.ops.aten._assert_tensor_metadata.default"
    sort_op = "torch.ops.aten.sort.default"
    aliases = {sort_op: ("torch.sort",)}

    unmatched = checker._find_unmatched_export_operators(
        [metadata_op, sort_op],
        set(),
        lambda op: aliases[op],
    )

    assert unmatched == [
        checker.UnmatchedExportOperator(
            exported_op=sort_op,
            public_apis=("torch.sort",),
        )
    ]


def test_old_unsupported_helper_remains_compatible() -> None:
    aliases = {"torch.ops.aten.sort.default": ("torch.sort",)}

    assert checker._find_unsupported_operators(
        aliases,
        set(),
        lambda op: aliases[op],
    ) == checker._find_unmatched_export_operators(
        aliases,
        set(),
        lambda op: aliases[op],
    )
    assert checker.UnsupportedOperator is checker.UnmatchedExportOperator


def test_checker_reuses_docgen_alias_mapping() -> None:
    resolver = checker._load_api_alias_resolver(REPO_ROOT)

    assert "torch.add" in resolver("torch.ops.aten.add.Tensor")
    assert "torch.sort" in resolver("torch.ops.aten.sort.default")
