#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Shared operator discovery and guard contract for the CMSIS Pack generators.

The kernel-registration generator (generate_register_all_kernels.py) wraps each
operator in ``#ifdef RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>``, and the component
generator (generate_components.py) emits the matching
``#define RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>`` in a PDSC component. The two
must use the identical token and cover the same set of operators -- otherwise a
selected component silently fails to register, or a registration can never be
enabled. This module is the single source of truth for both the guard naming and
the operator discovery so the two generators cannot drift.

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Display category -> source subdirectory (relative to the staged source root).
# Operators are discovered by scanning op_*.cpp in these directories.
CATEGORY_SUBDIRS: dict[str, str] = {
    "Portable": "kernels/portable/cpu",
    "Quantized": "kernels/quantized/cpu",
    "Cortex-M": "backends/cortex_m/ops",
}

# A kernel symbol's defining op_*.cpp is not derivable from its operator name
# (quantize_per_tensor.out lives in op_quantize.cpp), so the source file is
# found by scanning each op_*.cpp for the kernel functions it defines. Two
# forms occur: ordinary "ReturnType name(...) {" definitions and the
# DEFINE_*UFUNC*(name, ...) macros used by the elementwise unary ops. Only the
# names are needed -- the signatures all come from the YAML schema downstream.
_DEFINITION_RE = re.compile(
    r"""
    (?:^|\n)\s*
    (?:(?:::)?std::tuple<[^>]+>|Tensor&?|void)\s*   # return type
    \n?\s*
    (\w+)\s*\(                                   # kernel function name
    [^)]*(?:\([^)]*\)[^)]*)*?                    # parameters (tolerating nesting)
    \)\s*\{                                       # body open brace
    """,
    re.VERBOSE | re.MULTILINE,
)
_MACRO_RE = re.compile(r"DEFINE_\w*UFUNC\w*\s*\(\s*(\w+)\s*,")


def category_to_id(category: str) -> str:
    """Normalize a display category ("Cortex-M") to an identifier-safe token
    ("cortex_m") valid inside CMSIS condition IDs and C macros.
    """
    return category.lower().replace("-", "_").replace(" ", "_")


def op_to_guard(source_base: str, category: str) -> str:
    """Build the ``RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>`` guard for a source
    base and category.

    Leading underscores are stripped so an op name and its op_*.cpp filename
    always map to the same guard.

    """
    name = source_base.lstrip("_").upper()
    return f"RTE_ML_EXECUTORCH_OP_{category_to_id(category).upper()}_{name}"


@dataclass(frozen=True)
class Component:
    """One operator component == one op_*.cpp file == one guard."""

    name: str  # display/component name, e.g. "adaptive_avg_pool2d"
    category: str  # "Portable" | "Quantized" | "Cortex-M"
    source_file: Path  # path to op_*.cpp relative to the source root
    guard: str  # RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>


def _category_dir(source_dir: Path, subdir: str) -> Path | None:
    """Resolve a category subdir under the staged "src/" layout or the flat
    one.
    """
    for base in (source_dir / "src", source_dir):
        candidate = base / subdir
        if candidate.is_dir():
            return candidate
    return None


def discover_components(source_dir: Path) -> list[Component]:
    """Discover one Component per op_*.cpp under each category's source dir.

    The single source of truth for the operator set and guard names shared by
    both CMSIS Pack generators.

    """
    components: list[Component] = []
    for category, subdir in CATEGORY_SUBDIRS.items():
        op_dir = _category_dir(source_dir, subdir)
        if op_dir is None:
            continue
        for cpp in sorted(op_dir.glob("op_*.cpp")):
            base = cpp.stem[len("op_") :]
            components.append(
                Component(
                    name=base.lstrip("_"),
                    category=category,
                    source_file=cpp.relative_to(source_dir),
                    guard=op_to_guard(base, category),
                )
            )
    return components


def _defined_kernels(text: str) -> set[str]:
    """Names of the kernel functions defined in one op_*.cpp source."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return set(_DEFINITION_RE.findall(text)) | set(_MACRO_RE.findall(text))


def kernel_source_index(
    source_dir: Path, components: list[Component]
) -> dict[tuple[str, str], Component]:
    """Map (category, kernel function name) -> the Component (op_*.cpp) that
    defines it.

    This is how the registration generator finds the #ifdef guard for a kernel
    whose source file is not derivable from its operator name.

    """
    index: dict[tuple[str, str], Component] = {}
    for component in components:
        try:
            text = (source_dir / component.source_file).read_text(errors="ignore")
        except OSError:
            continue
        for name in _defined_kernels(text):
            index.setdefault((component.category, name), component)
    return index
