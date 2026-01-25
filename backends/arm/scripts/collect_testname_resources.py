# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ast
import pathlib
import re

from executorch.exir.dialects.edge.spec.utils import SAMPLE_INPUT as _SAMPLE_INPUT

# Add all targets and TOSA profiles we support here.
TARGETS = [
    "tosa_FP",
    "tosa_INT",
    "tosa_INT+FP",
    "u55_INT",
    "u85_INT",
    "vgf_INT",
    "vgf_FP",
    "vgf_quant",
    "vgf_no_quant",
    "no_target",
]

# Add edge ops which we lower but which are not included in exir/dialects/edge/edge.yaml here.
_CUSTOM_EDGE_OPS = [
    "linspace.default",
    "cond.default",
    "eye.default",
    "expm1.default",
    "gather.default",
    "vector_norm.default",
    "hardsigmoid.default",
    "hardswish.default",
    "linear.default",
    "maximum.default",
    "mean.default",
    "multihead_attention.default",
    "adaptive_avg_pool2d.default",
    "bitwise_right_shift.Tensor",
    "bitwise_right_shift.Scalar",
    "bitwise_left_shift.Tensor",
    "bitwise_left_shift.Scalar",
    "native_group_norm.default",
    "silu.default",
    "sdpa.default",
    "sum.default",
    "unbind.int",
    "unflatten.int",
    "unfold_copy.default",
    "_native_batch_norm_legit_no_training.default",
    "_native_batch_norm_legit.no_stats",
    "alias_copy.default",
    "pixel_shuffle.default",
    "pixel_unshuffle.default",
    "while_loop.default",
    "matmul.default",
    "upsample_bilinear2d.vec",
    "upsample_nearest2d.vec",
    "index_put.default",
]
_ALL_EDGE_OPS = _SAMPLE_INPUT.keys() | _CUSTOM_EDGE_OPS

_NON_ARM_PASSES = ["quantize_io_pass"]

_MODEL_ENTRY_PATTERN = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(?P<entry>.+?)\s*$")
_NUMERIC_SERIES_PATTERN = re.compile(r"(\d+)(?=[a-z])")
_CAMEL_BOUNDARY = re.compile(
    r"(?<!^)(?:(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))"
)


def _collect_arm_passes(init_path: pathlib.Path) -> set[str]:
    names: set[str] = set()
    names.update(_extract_pass_names_from_init(init_path))
    names.update(_NON_ARM_PASSES)
    return {_separate_numeric_series(_strip_pass_suffix(name)) for name in names}


def _extract_pass_names_from_init(init_path: pathlib.Path) -> set[str]:
    source = init_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(init_path))
    names: set[str] = set()

    for node in module.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            candidate = alias.asname or alias.name
            if not candidate or not candidate.endswith("Pass"):
                continue
            if candidate == "ArmPass":
                continue
            names.add(_camel_to_snake(candidate))
    return names


def _strip_pass_suffix(name: str) -> str:
    return name[:-5] if name.endswith("_pass") else name


def _separate_numeric_series(name: str) -> str:
    def repl(match: re.Match[str]) -> str:
        next_index = match.end()
        next_char = match.string[next_index] if next_index < len(match.string) else ""
        if next_char == "d":  # Avoid creating patterns like 3_d
            return match.group(1)
        return f"{match.group(1)}_"

    return _NUMERIC_SERIES_PATTERN.sub(repl, name)


def _collect_arm_models(models_md: pathlib.Path) -> set[str]:
    models: set[str] = set()
    for line in models_md.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _MODEL_ENTRY_PATTERN.match(line)
        if not match:
            continue
        base, alias, is_parent = _split_model_entry(match.group("entry"))
        if is_parent:
            continue
        if alias:
            models.add(_normalize_model_entry(alias))
        else:
            models.add(_normalize_model_entry(base))

    if not models:
        raise RuntimeError(f"No supported models found in {models_md}")
    return models


def _normalize_op_name(edge_name: str) -> str:
    op, overload = edge_name.split(".")

    op = op.lower()
    op = op.removeprefix("_")
    op = op.removesuffix("_copy")
    op = op.removesuffix("_with_indices")

    overload = overload.lower()
    if overload == "default":
        return op
    else:
        return f"{op}_{overload}"


def _split_model_entry(entry: str) -> tuple[str, str | None, bool]:
    entry = entry.strip()
    if not entry:
        return "", None, False
    is_parent = entry.endswith(":")
    if is_parent:
        entry = entry[:-1].rstrip()
    if "(" in entry and entry.endswith(")"):
        base, _, rest = entry.partition("(")
        alias = rest[:-1].strip()
        return base.strip(), alias or None, is_parent
    return entry, None, is_parent


def _normalize_model_entry(name: str) -> str:
    cleaned = name.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.replace(" ", "_")


def _camel_to_snake(name: str) -> str:
    if not name:
        return ""
    name = name.replace("-", "_").replace(" ", "_")
    return _CAMEL_BOUNDARY.sub("_", name).lower()


OP_NAME_MAP = {_normalize_op_name(edge_name): edge_name for edge_name in _ALL_EDGE_OPS}
OP_LIST = sorted({_normalize_op_name(edge_name) for edge_name in _ALL_EDGE_OPS})
PASS_LIST = sorted(
    _collect_arm_passes(pathlib.Path("backends/arm/_passes/__init__.py"))
)
MODEL_LIST = sorted(_collect_arm_models(pathlib.Path("backends/arm/MODELS.md")))
