#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Code generator for MLX delegate.

This is the SINGLE SOURCE OF TRUTH generator. Edit schema.fbs, then run:
    python generate.py

Generates:
1. FlatBuffer bindings (via flatc):
   - _generated/ (Python)
   - ../runtime/schema_generated.h (C++)
2. mlx_graph_schema.py (Python dataclasses)
3. _generated_serializers.py (Python serialization code)
4. ../runtime/MLXLoader.h (C++ structs, enums) - PARTIAL
5. ../runtime/MLXLoader.cpp (C++ loader switch) - PARTIAL

Usage:
    python generate.py [--flatc PATH_TO_FLATC] [--skip-flatc]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SCHEMA_FBS = SCRIPT_DIR / "schema.fbs"
GENERATED_DIR = SCRIPT_DIR / "_generated"
GENERATED_SERIALIZERS = SCRIPT_DIR / "_generated_serializers.py"
GENERATED_SCHEMA_PY = SCRIPT_DIR / "mlx_graph_schema.py"
GENERATED_INSPECTOR = SCRIPT_DIR.parent / "_generated_inspector.py"
RUNTIME_DIR = SCRIPT_DIR.parent / "runtime"
LOADER_H = RUNTIME_DIR / "MLXLoader.h"
LOADER_CPP = RUNTIME_DIR / "MLXLoader.cpp"


# =============================================================================
# FBS Parser
# =============================================================================


@dataclass
class FBSEnum:
    name: str
    base_type: str  # e.g., "byte"
    values: List[Tuple[str, Optional[int]]]  # (name, explicit_value or None)


@dataclass
class FBSField:
    name: str
    type_str: str  # e.g., "Tid", "[int32]", "IntOrVid"
    required: bool
    default: Optional[str]  # default value if specified


@dataclass
class FBSStruct:
    name: str
    fields: List[FBSField]


@dataclass
class FBSTable:
    name: str
    fields: List[FBSField]


@dataclass
class FBSUnion:
    name: str
    types: List[str]


@dataclass
class FBSSchema:
    namespace: str
    enums: List[FBSEnum]
    structs: List[FBSStruct]
    tables: List[FBSTable]
    unions: List[FBSUnion]

    def get_op_nodes(self) -> List[FBSTable]:
        """Get all tables that are part of the OpNode union."""
        op_union = next((u for u in self.unions if u.name == "OpNode"), None)
        if not op_union:
            return []
        op_names = set(op_union.types)
        return [t for t in self.tables if t.name in op_names]


def parse_fbs(fbs_path: Path) -> FBSSchema:
    """Parse a FlatBuffer schema file."""
    with open(fbs_path) as f:
        content = f.read()

    # Remove comments
    content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)

    namespace = ""
    enums: List[FBSEnum] = []
    structs: List[FBSStruct] = []
    tables: List[FBSTable] = []
    unions: List[FBSUnion] = []

    # Parse namespace
    ns_match = re.search(r"namespace\s+(\w+)\s*;", content)
    if ns_match:
        namespace = ns_match.group(1)

    # Parse enums
    for match in re.finditer(r"enum\s+(\w+)\s*:\s*(\w+)\s*\{([^}]+)\}", content):
        enum_name = match.group(1)
        base_type = match.group(2)
        body = match.group(3)
        values = []
        for val_match in re.finditer(r"(\w+)\s*(?:=\s*(\d+))?", body):
            name = val_match.group(1)
            explicit_val = int(val_match.group(2)) if val_match.group(2) else None
            values.append((name, explicit_val))
        enums.append(FBSEnum(enum_name, base_type, values))

    # Parse structs
    for match in re.finditer(r"struct\s+(\w+)\s*\{([^}]+)\}", content):
        struct_name = match.group(1)
        body = match.group(2)
        fields = _parse_fields(body)
        structs.append(FBSStruct(struct_name, fields))

    # Parse tables
    for match in re.finditer(r"table\s+(\w+)\s*\{([^}]*)\}", content):
        table_name = match.group(1)
        body = match.group(2)
        fields = _parse_fields(body)
        tables.append(FBSTable(table_name, fields))

    # Parse unions
    for match in re.finditer(r"union\s+(\w+)\s*\{([^}]+)\}", content):
        union_name = match.group(1)
        body = match.group(2)
        types = [t.strip() for t in body.split(",") if t.strip()]
        unions.append(FBSUnion(union_name, types))

    return FBSSchema(namespace, enums, structs, tables, unions)


def _parse_fields(body: str) -> List[FBSField]:
    """Parse fields from a struct/table body."""
    fields = []
    for line in body.split(";"):
        line = line.strip()
        if not line:
            continue

        # Parse: name: type (attributes) = default
        match = re.match(
            r"(\w+)\s*:\s*(\[?\w+\]?)\s*(?:\(([^)]*)\))?\s*(?:=\s*([^;]+))?", line
        )
        if match:
            name = match.group(1)
            type_str = match.group(2)
            attrs = match.group(3) or ""
            default = match.group(4).strip() if match.group(4) else None
            required = "required" in attrs
            fields.append(FBSField(name, type_str, required, default))

    return fields


# =============================================================================
# Python dataclass generation
# =============================================================================


def generate_python_schema(schema: FBSSchema) -> str:  # noqa: C901
    """Generate mlx_graph_schema.py from parsed FBS."""
    lines = [
        "#",
        "# Copyright (c) Meta Platforms, Inc. and affiliates.",
        "# All rights reserved.",
        "#",
        "# This source code is licensed under the BSD-style license found in the",
        "# LICENSE file in the root directory of this source tree.",
        "#",
        "# ============================================================================",
        "# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        "# ============================================================================",
        "#",
        "# This file was generated from schema.fbs by the MLX delegate code generator.",
        "#",
        "# Source:    backends/apple/mlx/serialization/schema.fbs",
        "# Generator: backends/apple/mlx/serialization/generate.py",
        "#",
        "# To regenerate, run from the executorch root:",
        "#     python backends/apple/mlx/serialization/generate.py",
        "#",
        "# ============================================================================",
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass, field",
        "from enum import IntEnum",
        "from typing import List, Optional, Union",
        "",
        "",
        "# =============================================================================",
        "# Enums",
        "# =============================================================================",
        "",
    ]

    # Generate enums
    for enum in schema.enums:
        lines.append(f"class {enum.name}(IntEnum):")
        val = 0
        for name, explicit_val in enum.values:
            if explicit_val is not None:
                val = explicit_val
            lines.append(f"    {name} = {val}")
            val += 1
        lines.append("")
        lines.append("")

    lines.extend(
        [
            "# =============================================================================",
            "# Core types",
            "# =============================================================================",
            "",
        ]
    )

    # Generate structs (Tid, Vid)
    for struct in schema.structs:
        lines.append("@dataclass")
        lines.append(f"class {struct.name}:")
        for fld in struct.fields:
            py_type = _fbs_type_to_python(fld.type_str, fld.required)
            default = _fbs_default_to_python(fld.default, fld.type_str)
            if default:
                lines.append(f"    {fld.name}: {py_type} = {default}")
            else:
                lines.append(f"    {fld.name}: {py_type}")
        lines.append("")
        lines.append("")

    # Generate IntOrVid with helper methods
    lines.extend(
        [
            "@dataclass",
            "class IntOrVid:",
            '    """Represents either a literal int or a runtime Vid reference."""',
            "    literal: int = 0",
            "    vid: Optional[Vid] = None",
            "    is_vid: bool = False",
            "",
            "    @classmethod",
            '    def from_literal(cls, value: int) -> "IntOrVid":',
            '        """Create an IntOrVid from a literal integer."""',
            "        return cls(literal=value, is_vid=False)",
            "",
            "    @classmethod",
            '    def from_vid(cls, vid: Vid) -> "IntOrVid":',
            '        """Create an IntOrVid from a Vid reference."""',
            "        return cls(vid=vid, is_vid=True)",
            "",
            "",
            "@dataclass",
            "class FloatOrVid:",
            '    """Represents either a literal float or a runtime Vid reference."""',
            "    literal: float = 0.0",
            "    vid: Optional[Vid] = None",
            "    is_vid: bool = False",
            "",
            "    @classmethod",
            '    def from_literal(cls, value: float) -> "FloatOrVid":',
            '        """Create a FloatOrVid from a literal float."""',
            "        return cls(literal=value, is_vid=False)",
            "",
            "    @classmethod",
            '    def from_vid(cls, vid: Vid) -> "FloatOrVid":',
            '        """Create a FloatOrVid from a Vid reference."""',
            "        return cls(vid=vid, is_vid=True)",
            "",
            "",
        ]
    )

    # Generate SlotVariant, NamedSlot, DataSegment, TensorMeta (but not Instruction/MLXGraph yet - they reference OpNode)
    other_tables = ["SlotVariant", "NamedSlot", "DataSegment", "TensorMeta"]
    for table_name in other_tables:
        table = next((t for t in schema.tables if t.name == table_name), None)
        if table:
            lines.append("@dataclass")
            lines.append(f"class {table.name}:")
            if not table.fields:
                lines.append("    pass")
            else:
                for fld in table.fields:
                    py_type = _fbs_type_to_python(fld.type_str, fld.required)
                    default = _fbs_default_to_python(fld.default, fld.type_str)
                    if default is not None:
                        lines.append(f"    {fld.name}: {py_type} = {default}")
                    elif not fld.required:
                        lines.append(f"    {fld.name}: {py_type} = None")
                    else:
                        lines.append(f"    {fld.name}: {py_type}")
            lines.append("")
            lines.append("")

    lines.extend(
        [
            "# =============================================================================",
            "# Op nodes",
            "# =============================================================================",
            "",
        ]
    )

    # Generate op node dataclasses
    op_nodes = schema.get_op_nodes()
    for table in op_nodes:
        lines.append("@dataclass")
        lines.append(f"class {table.name}:")
        if not table.fields:
            lines.append("    pass")
        else:
            # Separate required and optional fields (required first for dataclass)
            required_fields = []
            optional_fields = []
            for fld in table.fields:
                if fld.name.endswith("_is_set"):
                    continue  # Skip sentinel fields
                if fld.required or fld.default is not None:
                    required_fields.append(fld)
                else:
                    optional_fields.append(fld)

            for fld in required_fields:
                py_type = _fbs_type_to_python(fld.type_str, fld.required)
                default = _fbs_default_to_python(fld.default, fld.type_str)
                if default is not None:
                    lines.append(f"    {fld.name}: {py_type} = {default}")
                else:
                    lines.append(f"    {fld.name}: {py_type}")

            for fld in optional_fields:
                py_type = _fbs_type_to_python(fld.type_str, fld.required)
                lines.append(f"    {fld.name}: {py_type} = None")

        lines.append("")
        lines.append("")

    # Generate OpNodeUnion type alias
    op_names = [t.name for t in op_nodes]
    lines.append("# Union of all op types")
    lines.append("OpNodeUnion = Union[")
    for name in op_names:
        lines.append(f"    {name},")
    lines.append("]")
    lines.append("")

    # Generate Instruction and MLXGraph (these reference OpNode so must come after)
    lines.extend(
        [
            "# =============================================================================",
            "# Container types (reference OpNodeUnion)",
            "# =============================================================================",
            "",
            "@dataclass",
            "class Instruction:",
            "    op: OpNodeUnion",
            "",
            "",
            "@dataclass",
            "class MLXGraph:",
            "    instructions: List[Instruction]",
            "    version: Optional[str] = None",
            "    num_constant_tensors: int = 0",
            "    num_non_constant_tensors: int = 0",
            "    num_non_constant_values: int = 0",
            "    input_map: Optional[List[SlotVariant]] = None",
            "    output_map: Optional[List[SlotVariant]] = None",
            "    mutable_buffer_map: Optional[List[SlotVariant]] = None",
            "    named_slots: Optional[List[NamedSlot]] = None",
            "    tensor_meta: Optional[List[TensorMeta]] = None",
            "    constant_segment: Optional[DataSegment] = None",
            "",
        ]
    )

    return "\n".join(lines)


def _fbs_type_to_python(fbs_type: str, required: bool) -> str:
    """Convert FBS type to Python type annotation."""
    # Handle arrays
    if fbs_type.startswith("[") and fbs_type.endswith("]"):
        inner = fbs_type[1:-1]
        inner_py = _fbs_type_to_python(inner, True)
        return f"List[{inner_py}]"

    # Map FBS types to Python
    type_map = {
        "int32": "int",
        "int64": "int",
        "uint32": "int",
        "uint64": "int",
        "float": "float",
        "double": "float",
        "bool": "bool",
        "string": "str",
        "byte": "int",
    }

    py_type = type_map.get(fbs_type, fbs_type)

    if not required and fbs_type not in type_map:
        # Optional reference types
        return f"Optional[{py_type}]"

    return py_type


def _fbs_default_to_python(default: Optional[str], fbs_type: str) -> Optional[str]:
    """Convert FBS default value to Python."""
    if default is None:
        return None

    if default == "false":
        return "False"
    if default == "true":
        return "True"
    if default == "null":
        return "None"

    # Handle enum defaults like 'TensorSlot'
    if fbs_type == "SlotType":
        return f"SlotType.{default}"

    # Numeric defaults
    return default


# =============================================================================
# Python serializer generation (existing code, refactored)
# =============================================================================


def generate_python_serializers(schema: FBSSchema) -> str:
    """Generate _generated_serializers.py from parsed FBS."""
    op_nodes = schema.get_op_nodes()

    # Generate op type names dict from union order
    op_union = next((u for u in schema.unions if u.name == "OpNode"), None)
    op_type_names_lines = [
        "# FlatBuffer union indices: 0 = NONE, then 1-indexed from union order"
    ]
    op_type_names_lines.append("MLX_OP_TYPE_NAMES = {")
    op_type_names_lines.append('    0: "NONE",')
    if op_union:
        for i, type_name in enumerate(op_union.types, start=1):
            op_type_names_lines.append(f'    {i}: "{type_name}",')
    op_type_names_lines.append("}")
    op_type_names = "\n".join(op_type_names_lines)

    # Build imports
    op_imports = ",\n".join(f"    {t.name}" for t in op_nodes)

    header = f'''#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# ============================================================================
# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
# ============================================================================
#
# This file was generated from schema.fbs by the MLX delegate code generator.
#
# Source:    backends/apple/mlx/serialization/schema.fbs
# Generator: backends/apple/mlx/serialization/generate.py
#
# To regenerate, run from the executorch root:
#     python backends/apple/mlx/serialization/generate.py
#
# ============================================================================
#
# This file contains auto-generated serializer methods for all op types.

from __future__ import annotations

from typing import List, Tuple, Dict

import flatbuffers

{op_type_names}

from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (
{op_imports},
    IntOrVid,
    FloatOrVid,
    Tid,
    Vid,
)


def _build_int_vector(builder: flatbuffers.Builder, vec: List[int]) -> int:
    """Build a vector of int32."""
    builder.StartVector(4, len(vec), 4)
    for v in reversed(vec):
        builder.PrependInt32(v)
    return builder.EndVector()


class GeneratedOpBuilders:
    """Mixin class with auto-generated op builder methods."""

    def _build_int_or_vid(self, builder: flatbuffers.Builder, iov: IntOrVid) -> int:
        """Build an IntOrVid table."""
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import IntOrVid as FBIntOrVidModule
        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid

        FBIntOrVidModule.Start(builder)
        FBIntOrVidModule.AddLiteral(builder, iov.literal)
        FBIntOrVidModule.AddIsVid(builder, iov.is_vid)
        if iov.vid is not None:
            # Vid is an inline struct - must be added last for proper FlatBuffer layout
            FBIntOrVidModule.AddVid(builder, CreateVid(builder, iov.vid.idx))
        return FBIntOrVidModule.End(builder)

    def _build_int_or_vid_vector(
        self, builder: flatbuffers.Builder, vec: List[IntOrVid]
    ) -> int:
        """Build a vector of IntOrVid tables."""
        offsets = []
        for iov in vec:
            offsets.append(self._build_int_or_vid(builder, iov))
        builder.StartVector(4, len(offsets), 4)
        for off in reversed(offsets):
            builder.PrependUOffsetTRelative(off)
        return builder.EndVector()

'''

    # Generate builder methods for each op
    methods = []
    for table in op_nodes:
        methods.append(_generate_op_builder_method(table))

    return header + "\n".join(methods)


def _generate_op_builder_method(table: FBSTable) -> str:  # noqa: C901
    """Generate a _build_XxxNode method for the serializer class."""
    class_name = table.name
    fb_module_name = f"FB{class_name}Module"

    lines = [
        f"    def _build_{class_name}(",
        f"        self, builder: flatbuffers.Builder, op: {class_name}",
        "    ) -> Tuple[int, int]:",
        f'        """Auto-generated builder for {class_name}."""',
        "        # Import the MODULE (not class) to access builder functions like Start(), Add*(), End()",
        f"        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import {class_name} as {fb_module_name}",
        "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import OpNode as FBOpNodeModule",
        "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid",
        "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid",
        "",
    ]

    # Pre-build any strings or vectors (must be done before Start)
    prebuild_lines = []
    for fld in table.fields:
        if fld.name.endswith("_is_set"):
            continue
        kind = _get_field_kind(fld, table)

        if kind == "str":
            prebuild_lines.append(
                f"        {fld.name}_off = builder.CreateString(op.{fld.name})"
            )
        elif kind == "list_int":
            prebuild_lines.append(
                f"        {fld.name}_vec = _build_int_vector(builder, op.{fld.name})"
            )
        elif kind == "list_int_or_vid":
            prebuild_lines.append(
                f"        {fld.name}_vec = self._build_int_or_vid_vector(builder, op.{fld.name})"
            )
        elif kind == "int_or_vid":
            prebuild_lines.append(
                f"        {fld.name}_off = self._build_int_or_vid(builder, op.{fld.name})"
            )
        elif kind == "optional_str":
            prebuild_lines.append(
                f"        {fld.name}_off = builder.CreateString(op.{fld.name}) if op.{fld.name} is not None else None"
            )

    if prebuild_lines:
        lines.extend(prebuild_lines)
        lines.append("")

    # Start the FlatBuffer table
    lines.append(f"        {fb_module_name}.Start(builder)")

    # Add each field
    for fld in table.fields:
        if fld.name.endswith("_is_set"):
            continue
        fb_field_name = _to_pascal_case(fld.name)
        kind = _get_field_kind(fld, table)

        if kind == "tid":
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, CreateTid(builder, op.{fld.name}.idx))"
            )
        elif kind == "vid":
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, CreateVid(builder, op.{fld.name}.idx))"
            )
        elif kind in ("int", "float", "bool"):
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, op.{fld.name})"
            )
        elif kind == "str":
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, {fld.name}_off)"
            )
        elif kind == "dtype":
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, op.{fld.name})"
            )
        elif kind == "list_int":
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, {fld.name}_vec)"
            )
        elif kind == "list_int_or_vid":
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, {fld.name}_vec)"
            )
        elif kind == "int_or_vid":
            lines.append(
                f"        {fb_module_name}.Add{fb_field_name}(builder, {fld.name}_off)"
            )
        elif kind == "optional_tid":
            lines.append(f"        if op.{fld.name} is not None:")
            lines.append(
                f"            {fb_module_name}.Add{fb_field_name}(builder, CreateTid(builder, op.{fld.name}.idx))"
            )
        elif kind == "optional_vid":
            lines.append(f"        if op.{fld.name} is not None:")
            lines.append(
                f"            {fb_module_name}.Add{fb_field_name}(builder, CreateVid(builder, op.{fld.name}.idx))"
            )
        elif kind == "optional_float":
            lines.append(f"        if op.{fld.name} is not None:")
            lines.append(
                f"            {fb_module_name}.Add{fb_field_name}(builder, op.{fld.name})"
            )
        elif kind == "optional_dtype":
            lines.append(f"        if op.{fld.name} is not None:")
            lines.append(
                f"            {fb_module_name}.Add{fb_field_name}(builder, op.{fld.name})"
            )
        elif kind == "optional_str":
            lines.append(f"        if {fld.name}_off is not None:")
            lines.append(
                f"            {fb_module_name}.Add{fb_field_name}(builder, {fld.name}_off)"
            )

    # End the FlatBuffer table and return offset + union type
    lines.append(f"        offset = {fb_module_name}.End(builder)")
    lines.append(f"        return offset, FBOpNodeModule.OpNode.{class_name}")
    lines.append("")

    return "\n".join(lines)


def _get_field_kind(fld: FBSField, table: FBSTable) -> str:  # noqa: C901
    """Determine the kind of a field for serialization."""
    t = fld.type_str

    # Handle arrays
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1]
        if inner in ("int32", "int64"):
            return "list_int"
        if inner == "IntOrVid":
            return "list_int_or_vid"
        return f"list_{inner}"

    # Handle basic types
    if t == "Tid":
        return "optional_tid" if not fld.required else "tid"
    if t == "Vid":
        return "optional_vid" if not fld.required else "vid"
    if t == "IntOrVid":
        return "int_or_vid"
    if t == "FloatOrVid":
        return "float_or_vid"
    if t == "DTypeId":
        # Check if this is optional (has = null default)
        if fld.default == "null":
            return "optional_dtype"
        return "dtype"
    if t in ("int32", "int64", "uint32", "uint64"):
        return "int"
    if t in ("float", "double"):
        # Check if this is optional (has = null default)
        if fld.default == "null":
            return "optional_float"
        return "float"
    if t == "bool":
        return "bool"
    if t == "string":
        return "optional_str" if not fld.required else "str"

    return "unknown"


def _to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    # Handle special cases
    if name == "table_":
        return "Table_"
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts)


# =============================================================================
# C++ generation
# =============================================================================


def generate_cpp_loader_h(schema: FBSSchema) -> str:
    """Generate MLXLoader.h from parsed FBS."""
    op_nodes = schema.get_op_nodes()

    lines = [
        "//",
        "// Copyright (c) Meta Platforms, Inc. and affiliates.",
        "// All rights reserved.",
        "//",
        "// This source code is licensed under the BSD-style license found in the",
        "// LICENSE file in the root directory of this source tree.",
        "//",
        "// ============================================================================",
        "// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        "// ============================================================================",
        "//",
        "// This file was generated from schema.fbs by the MLX delegate code generator.",
        "//",
        "// Source:    backends/apple/mlx/serialization/schema.fbs",
        "// Generator: backends/apple/mlx/serialization/generate.py",
        "//",
        "// To regenerate, run from the executorch root:",
        "//     python backends/apple/mlx/serialization/generate.py",
        "//",
        "// ============================================================================",
        "//",
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <cstring>",
        "#include <optional>",
        "#include <stdexcept>",
        "#include <string>",
        "#include <variant>",
        "#include <vector>",
        "",
        '#include "schema_generated.h"',
        "",
        "namespace executorch {",
        "namespace backends {",
        "namespace mlx {",
        "",
        "// =============================================================================",
        "// Core types matching the Python side",
        "// =============================================================================",
        "",
        "struct Tid {",
        "  uint32_t idx{};",
        "};",
        "",
        "template <typename T>",
        "struct Vid {",
        "  uint32_t idx{};",
        "};",
        "",
    ]

    # Generate DTypeId enum
    dtype_enum = next((e for e in schema.enums if e.name == "DTypeId"), None)
    if dtype_enum:
        lines.append("enum class DTypeId : int {")
        for name, _val in dtype_enum.values:
            lines.append(f"  {name},")
        lines.append("};")
        lines.append("")

    lines.extend(
        [
            "// =============================================================================",
            "// Tensor metadata",
            "// =============================================================================",
            "",
            "struct TensorMeta {",
            "  std::vector<std::variant<int64_t, Vid<int32_t>>> shape;",
            "  DTypeId dtype;",
            "  std::vector<int32_t> strides;",
            "};",
            "",
            "// =============================================================================",
            "// Constant segment info",
            "// =============================================================================",
            "",
            "struct ConstantSegment {",
            "  uint64_t offset;",
            "  uint64_t size;",
            "};",
            "",
            "// =============================================================================",
            "// Op node types (AUTO-GENERATED from schema.fbs)",
            "// =============================================================================",
            "",
        ]
    )

    # Generate op node structs
    for table in op_nodes:
        lines.append(f"struct {table.name} {{")
        if not table.fields:
            lines.append("};")
        else:
            for fld in table.fields:
                if fld.name.endswith("_is_set"):
                    continue
                cpp_type = _fbs_type_to_cpp(fld.type_str, fld.required, table, fld)
                lines.append(f"  {cpp_type} {fld.name};")
            lines.append("};")
        lines.append("")

    # Generate OpCode enum
    lines.extend(
        [
            "// =============================================================================",
            "// OpCode enum (AUTO-GENERATED from schema.fbs)",
            "// =============================================================================",
            "",
            "enum class OpCode : uint8_t {",
        ]
    )

    for table in op_nodes:
        op_code = _table_name_to_opcode(table.name)
        lines.append(f"  {op_code},")
    lines.append("  SENTINEL")
    lines.append("};")
    lines.append("")

    # Generate NodeVariant
    lines.extend(
        [
            "// =============================================================================",
            "// NodeVariant for type-erased op storage (AUTO-GENERATED)",
            "// =============================================================================",
            "",
            "using NodeVariant = std::variant<",
        ]
    )
    for i, table in enumerate(op_nodes):
        comma = "," if i < len(op_nodes) - 1 else ">"
        lines.append(f"    {table.name}{comma}")
    lines.append(";")
    lines.append("")

    # Add the rest of the header (manual parts)
    lines.extend(
        [
            "// =============================================================================",
            "// Instruction",
            "// =============================================================================",
            "",
            "struct Instruction {",
            "  OpCode op{OpCode::NOOP};",
            "  NodeVariant node;",
            "",
            "  template <typename T>",
            "  T& get() {",
            "    return std::get<T>(node);",
            "  }",
            "",
            "  template <typename T>",
            "  const T& get() const {",
            "    return std::get<T>(node);",
            "  }",
            "};",
            "",
            "// =============================================================================",
            "// SlotVariant for I/O mapping",
            "// =============================================================================",
            "",
            "enum class SlotType : uint8_t {",
            "  TensorSlot = 0,",
            "  IntValueSlot = 1,",
            "  FloatValueSlot = 2,",
            "  BoolValueSlot = 3,",
            "};",
            "",
            "struct SlotVariant {",
            "  uint32_t idx;",
            "  SlotType slot_type;",
            "};",
            "",
            "// =============================================================================",
            "// Named slot (name -> slot mapping)",
            "// =============================================================================",
            "",
            "struct NamedSlot {",
            "  std::string name;",
            "  SlotVariant slot;",
            "};",
            "",
            "// =============================================================================",
            "// MLXProgram - the loaded program ready for execution",
            "// =============================================================================",
            "",
            "struct MLXProgram {",
            "  std::string version;",
            "",
            "  // Tensor/value slot counts",
            "  uint32_t num_constant_tensors{0};",
            "  uint32_t num_non_constant_tensors{0};",
            "  uint32_t num_non_constant_values{0};",
            "",
            "  // Instructions",
            "  std::vector<Instruction> instructions;",
            "",
            "  // I/O mappings",
            "  std::vector<SlotVariant> input_map;",
            "  std::vector<SlotVariant> output_map;",
            "  std::vector<SlotVariant> mutable_buffer_map;",
            "",
            "  // Name to slot lookup",
            "  std::vector<NamedSlot> named_slots;",
            "",
            "  // Tensor metadata",
            "  std::vector<std::optional<TensorMeta>> tensor_meta;",
            "",
            "  // Constant segment info",
            "  ConstantSegment constant_segment;",
            "",
            "  // Pointer to constant data (set after loading)",
            "  const uint8_t* constant_data{nullptr};",
            "",
            "  // Helper methods",
            "  inline uint32_t num_tensors() const {",
            "    return num_constant_tensors + num_non_constant_tensors;",
            "  }",
            "",
            "  inline uint32_t num_values() const {",
            "    return num_non_constant_values;",
            "  }",
            "",
            "  inline bool is_constant_tensor(Tid id) const {",
            "    return id.idx < num_constant_tensors;",
            "  }",
            "",
            "  inline size_t num_inputs() const {",
            "    return input_map.size();",
            "  }",
            "",
            "  inline size_t num_outputs() const {",
            "    return output_map.size();",
            "  }",
            "};",
            "",
            "// =============================================================================",
            "// FlatBuffer loading functions",
            "// =============================================================================",
            "",
            "namespace loader {",
            "",
        ]
    )

    # Generate convert_dtype
    if dtype_enum:
        lines.append("// Convert FlatBuffer DTypeId to our DTypeId")
        lines.append("inline DTypeId convert_dtype(mlx_delegate::DTypeId fb_dtype) {")
        lines.append("  switch (fb_dtype) {")
        for name, _ in dtype_enum.values:
            lines.append(f"    case mlx_delegate::DTypeId_{name}:")
            lines.append(f"      return DTypeId::{name};")
        lines.append("    default:")
        lines.append("      return DTypeId::f32;")
        lines.append("  }")
        lines.append("}")
        lines.append("")

    lines.extend(
        [
            "// Convert FlatBuffer SlotType to our SlotType",
            "inline SlotType convert_slot_type(mlx_delegate::SlotType fb_type) {",
            "  switch (fb_type) {",
            "    case mlx_delegate::SlotType_TensorSlot:",
            "      return SlotType::TensorSlot;",
            "    case mlx_delegate::SlotType_IntValueSlot:",
            "      return SlotType::IntValueSlot;",
            "    case mlx_delegate::SlotType_FloatValueSlot:",
            "      return SlotType::FloatValueSlot;",
            "    case mlx_delegate::SlotType_BoolValueSlot:",
            "      return SlotType::BoolValueSlot;",
            "    default:",
            "      return SlotType::TensorSlot;",
            "  }",
            "}",
            "",
            "// Convert FlatBuffer Tid",
            "inline Tid convert_tid(const mlx_delegate::Tid* fb_tid) {",
            "  if (!fb_tid) {",
            "    return Tid{0};",
            "  }",
            "  return Tid{fb_tid->idx()};",
            "}",
            "",
            "// Convert FlatBuffer Vid",
            "inline Vid<int32_t> convert_vid(const mlx_delegate::Vid* fb_vid) {",
            "  if (!fb_vid) {",
            "    return Vid<int32_t>{0};",
            "  }",
            "  return Vid<int32_t>{fb_vid->idx()};",
            "}",
            "",
            "// Convert FlatBuffer IntOrVid",
            "inline std::variant<int64_t, Vid<int32_t>> convert_int_or_vid(",
            "    const mlx_delegate::IntOrVid* fb) {",
            "  if (!fb) {",
            "    return int64_t{0};",
            "  }",
            "  if (!fb->is_vid()) {",
            "    return fb->literal();",
            "  }",
            "  const auto* vid_ptr = fb->vid();",
            "  if (!vid_ptr) {",
            "    return int64_t{0};",
            "  }",
            "  return Vid<int32_t>{vid_ptr->idx()};",
            "}",
            "",
            "// Convert FlatBuffer SlotVariant",
            "inline SlotVariant convert_slot_variant(const mlx_delegate::SlotVariant* fb) {",
            "  if (!fb) {",
            "    return SlotVariant{0, SlotType::TensorSlot};",
            "  }",
            "  return SlotVariant{fb->idx(), convert_slot_type(fb->slot_type())};",
            "}",
            "",
            "// Load an instruction from FlatBuffer",
            "Instruction load_instruction(const mlx_delegate::Instruction* fb_instr);",
            "",
            "// Load the full MLXProgram from FlatBuffer data",
            "MLXProgram load_program(const void* data, size_t size);",
            "",
            "} // namespace loader",
            "",
            "} // namespace mlx",
            "} // namespace backends",
            "} // namespace executorch",
        ]
    )

    return "\n".join(lines)


def _fbs_type_to_cpp(
    fbs_type: str,
    required: bool,
    table: Optional["FBSTable"] = None,
    fld: Optional["FBSField"] = None,
) -> str:
    """Convert FBS type to C++ type.

    Args:
        fbs_type: The FlatBuffer type string
        required: Whether the field is required
        table: Optional table context for type inference
        fld: Optional field context for the current field

    Note: Scalar types (float, int, DTypeId, etc.) are NEVER optional in C++.
    The Python serialization layer is responsible for ensuring all scalar fields
    have values (using defaults if user doesn't provide them).
    Only reference types (Tid, Vid) can be optional since they represent
    genuinely optional tensors/values.
    """
    # Handle arrays
    if fbs_type.startswith("[") and fbs_type.endswith("]"):
        inner = fbs_type[1:-1]
        inner_cpp = _fbs_type_to_cpp(inner, True)
        return f"std::vector<{inner_cpp}>"

    # Map FBS types to C++
    type_map = {
        "int32": "int32_t",
        "int64": "int64_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
        "float": "float",
        "double": "double",
        "bool": "bool",
        "string": "std::string",
        "byte": "uint8_t",
        "Tid": "Tid",
        "Vid": "Vid<int32_t>",
        "DTypeId": "DTypeId",
        "IntOrVid": "std::variant<int64_t, Vid<int32_t>>",
        "FloatOrVid": "std::variant<double, Vid<int32_t>>",
    }

    cpp_type = type_map.get(fbs_type, fbs_type)

    # Handle optional types - ONLY for reference types (Tid, Vid)
    # Scalar types are never optional; Python serializer ensures values are present
    if not required:
        if fbs_type == "Tid":
            return "std::optional<Tid>"
        if fbs_type == "Vid":
            return "std::optional<Vid<int32_t>>"

    return cpp_type


def _table_name_to_opcode(name: str) -> str:
    """Convert table name like 'LinearNode' to opcode like 'LINEAR'.

    Handles special cases like:
    - RMSNormNode -> RMS_NORM (not R_M_S_NORM)
    - Conv1DNode -> CONV1D (not CONV1_D)
    - ARangeNode -> ARANGE (not A_RANGE)
    - IdCopyNode -> ID_COPY
    """
    # Remove 'Node' suffix
    if name.endswith("Node"):
        name = name[:-4]

    # Special case mappings for acronyms and numbers
    special_cases = {
        "RMSNorm": "RMS_NORM",
        "LayerNorm": "LAYER_NORM",
        "Conv1D": "CONV1D",
        "ARange": "ARANGE",
        "IdCopy": "ID_COPY",
        "SymSize": "SYM_SIZE",
        "ItemInt": "ITEM_INT",
        "ExpandDims": "EXPAND_DIMS",
        "TakeAlongAxis": "TAKE_ALONG_AXIS",
        "AddScalar": "ADD_SCALAR",
        "SliceUpdate": "SLICE_UPDATE",
        "QuantizedLinear": "QUANTIZED_LINEAR",
        "QuantizedGather": "QUANTIZED_GATHER",
    }

    if name in special_cases:
        return special_cases[name]

    # For simple names, just uppercase
    # This handles: Noop, Linear, Tile, Rope, Sdpa, Add, Mul, Gelu, Silu,
    #               Reshape, Transpose, Contiguous, Gather, Slice, Cast,
    #               Concat, Full, Zeros, Ones, Argmax
    return name.upper()


def generate_cpp_loader_cpp(schema: FBSSchema) -> str:
    """Generate MLXLoader.cpp from parsed FBS."""
    op_nodes = schema.get_op_nodes()

    lines = [
        "//",
        "// Copyright (c) Meta Platforms, Inc. and affiliates.",
        "// All rights reserved.",
        "//",
        "// This source code is licensed under the BSD-style license found in the",
        "// LICENSE file in the root directory of this source tree.",
        "//",
        "// ============================================================================",
        "// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        "// ============================================================================",
        "//",
        "// This file was generated from schema.fbs by the MLX delegate code generator.",
        "//",
        "// Source:    backends/apple/mlx/serialization/schema.fbs",
        "// Generator: backends/apple/mlx/serialization/generate.py",
        "//",
        "// To regenerate, run from the executorch root:",
        "//     python backends/apple/mlx/serialization/generate.py",
        "//",
        "// ============================================================================",
        "//",
        "",
        '#include "MLXLoader.h"',
        "",
        "#include <cstring>",
        "#include <stdexcept>",
        "",
        "namespace executorch {",
        "namespace backends {",
        "namespace mlx {",
        "namespace loader {",
        "",
        "namespace {",
        "",
        "// Header structure for MLX payload",
        "constexpr size_t kHeaderSize = 24;",
        'constexpr uint32_t kMagic = 0x30584C4D;  // "MLX0" in little-endian',
        "",
        "struct MLXHeader {",
        "  uint32_t padding;",
        "  uint32_t magic;",
        "  uint64_t data_offset;",
        "  uint64_t data_size;",
        "};",
        "",
        "bool parse_header(const void* data, size_t size, MLXHeader& header) {",
        "  if (size < kHeaderSize) {",
        "    return false;",
        "  }",
        "  std::memcpy(&header, data, sizeof(MLXHeader));",
        "  if (header.magic != kMagic) {",
        "    return false;",
        "  }",
        "  return true;",
        "}",
        "",
        "// Helper to convert FlatBuffer vectors to std::vector",
        "template <typename T>",
        "std::vector<T> to_vector(const flatbuffers::Vector<T>* fb_vec) {",
        "  if (!fb_vec) {",
        "    return {};",
        "  }",
        "  return std::vector<T>(fb_vec->begin(), fb_vec->end());",
        "}",
        "",
        "}  // namespace",
        "",
        "// =============================================================================",
        "// load_instruction - AUTO-GENERATED switch statement",
        "// =============================================================================",
        "",
        "Instruction load_instruction(const mlx_delegate::Instruction* fb_instr) {",
        "  Instruction instr;",
        "",
        "  if (!fb_instr || !fb_instr->op()) {",
        "    instr.op = OpCode::NOOP;",
        "    instr.node = NoopNode{};",
        "    return instr;",
        "  }",
        "",
        "  auto op_type = fb_instr->op_type();",
        "",
        "  switch (op_type) {",
    ]

    # Generate switch cases for each op
    for table in op_nodes:
        lines.extend(_generate_loader_case(table))

    lines.extend(
        [
            "    default: {",
            "      instr.op = OpCode::NOOP;",
            "      instr.node = NoopNode{};",
            "      break;",
            "    }",
            "  }",
            "",
            "  return instr;",
            "}",
            "",
        ]
    )

    # Add load_program function (mostly static)
    lines.extend(
        [
            "// =============================================================================",
            "// load_program",
            "// =============================================================================",
            "",
            "MLXProgram load_program(const void* data, size_t size) {",
            "  MLXHeader header;",
            "  if (!parse_header(data, size, header)) {",
            '    throw std::runtime_error("Invalid MLX header");',
            "  }",
            "",
            "  const uint8_t* fb_data = static_cast<const uint8_t*>(data) + kHeaderSize;",
            "  size_t fb_size = header.data_offset - kHeaderSize;",
            "",
            "  flatbuffers::Verifier verifier(fb_data, fb_size);",
            "  if (!mlx_delegate::VerifyMLXGraphBuffer(verifier)) {",
            '    throw std::runtime_error("Invalid FlatBuffer data");',
            "  }",
            "",
            "  const auto* fb_graph = mlx_delegate::GetMLXGraph(fb_data);",
            "  if (!fb_graph) {",
            '    throw std::runtime_error("Failed to parse MLXGraph");',
            "  }",
            "",
            "  MLXProgram program;",
            "",
            "  if (fb_graph->version()) {",
            "    program.version = fb_graph->version()->str();",
            "  }",
            "",
            "  program.num_constant_tensors = fb_graph->num_constant_tensors();",
            "  program.num_non_constant_tensors = fb_graph->num_non_constant_tensors();",
            "  program.num_non_constant_values = fb_graph->num_non_constant_values();",
            "",
            "  if (fb_graph->instructions()) {",
            "    program.instructions.reserve(fb_graph->instructions()->size());",
            "    for (size_t i = 0; i < fb_graph->instructions()->size(); ++i) {",
            "      const auto* fb_instr = fb_graph->instructions()->Get(i);",
            "      program.instructions.push_back(load_instruction(fb_instr));",
            "    }",
            "  }",
            "",
            "  if (fb_graph->input_map()) {",
            "    for (size_t i = 0; i < fb_graph->input_map()->size(); ++i) {",
            "      const auto* slot = fb_graph->input_map()->Get(i);",
            "      program.input_map.push_back(convert_slot_variant(slot));",
            "    }",
            "  }",
            "",
            "  if (fb_graph->output_map()) {",
            "    for (size_t i = 0; i < fb_graph->output_map()->size(); ++i) {",
            "      const auto* slot = fb_graph->output_map()->Get(i);",
            "      program.output_map.push_back(convert_slot_variant(slot));",
            "    }",
            "  }",
            "",
            "  if (fb_graph->mutable_buffer_map()) {",
            "    for (size_t i = 0; i < fb_graph->mutable_buffer_map()->size(); ++i) {",
            "      const auto* slot = fb_graph->mutable_buffer_map()->Get(i);",
            "      program.mutable_buffer_map.push_back(convert_slot_variant(slot));",
            "    }",
            "  }",
            "",
            "  if (fb_graph->named_slots()) {",
            "    for (size_t i = 0; i < fb_graph->named_slots()->size(); ++i) {",
            "      const auto* fb_slot = fb_graph->named_slots()->Get(i);",
            "      NamedSlot slot;",
            '      slot.name = fb_slot->name() ? fb_slot->name()->str() : "";',
            "      slot.slot = convert_slot_variant(fb_slot->slot());",
            "      program.named_slots.push_back(std::move(slot));",
            "    }",
            "  }",
            "",
            "  if (fb_graph->tensor_meta()) {",
            "    for (size_t i = 0; i < fb_graph->tensor_meta()->size(); ++i) {",
            "      const auto* fb_meta = fb_graph->tensor_meta()->Get(i);",
            "      if (fb_meta) {",
            "        TensorMeta meta;",
            "        if (fb_meta->shape()) {",
            "          for (size_t j = 0; j < fb_meta->shape()->size(); ++j) {",
            "            const auto* iov = fb_meta->shape()->Get(j);",
            "            meta.shape.push_back(convert_int_or_vid(iov));",
            "          }",
            "        }",
            "        meta.dtype = convert_dtype(fb_meta->dtype());",
            "        meta.strides = to_vector(fb_meta->strides());",
            "        program.tensor_meta.push_back(std::move(meta));",
            "      } else {",
            "        program.tensor_meta.push_back(std::nullopt);",
            "      }",
            "    }",
            "  }",
            "",
            "  if (fb_graph->constant_segment()) {",
            "    program.constant_segment.offset = fb_graph->constant_segment()->offset();",
            "    program.constant_segment.size = fb_graph->constant_segment()->size();",
            "  }",
            "",
            "  program.constant_data =",
            "      static_cast<const uint8_t*>(data) + header.data_offset;",
            "",
            "  return program;",
            "}",
            "",
            "}  // namespace loader",
            "}  // namespace mlx",
            "}  // namespace backends",
            "}  // namespace executorch",
        ]
    )

    return "\n".join(lines)


def _generate_loader_case(table: FBSTable) -> List[str]:  # noqa: C901
    """Generate a switch case for loading an op node."""
    class_name = table.name
    op_code = _table_name_to_opcode(class_name)

    lines = [
        f"    case mlx_delegate::OpNode_{class_name}: {{",
    ]

    if not table.fields:
        # NoopNode case
        lines.extend(
            [
                f"      instr.op = OpCode::{op_code};",
                f"      instr.node = {class_name}{{}};",
                "      break;",
                "    }",
                "",
            ]
        )
        return lines

    lines.append(f"      auto fb = fb_instr->op_as_{class_name}();")
    lines.append(f"      {class_name} node;")

    for fld in table.fields:
        if fld.name.endswith("_is_set"):
            continue

        # FlatBuffer C++ accessor uses the original field name from the schema
        fb_field_name = fld.name
        kind = _get_field_kind(fld, table)

        if kind == "tid":
            lines.append(f"      node.{fld.name} = convert_tid(fb->{fb_field_name}());")
        elif kind == "optional_tid":
            lines.append(f"      if (fb->{fb_field_name}()) {{")
            lines.append(
                f"        node.{fld.name} = convert_tid(fb->{fb_field_name}());"
            )
            lines.append("      }")
        elif kind == "vid":
            lines.append(f"      node.{fld.name} = convert_vid(fb->{fb_field_name}());")
        elif kind == "optional_vid":
            lines.append(f"      if (fb->{fb_field_name}()) {{")
            lines.append(
                f"        node.{fld.name} = convert_vid(fb->{fb_field_name}());"
            )
            lines.append("      }")
        elif kind == "int":
            lines.append(f"      node.{fld.name} = fb->{fb_field_name}();")
        elif kind == "bool":
            lines.append(f"      node.{fld.name} = fb->{fb_field_name}();")
        elif kind == "float":
            lines.append(f"      node.{fld.name} = fb->{fb_field_name}();")
        elif kind == "optional_float":
            # Optional scalar with = null default - FlatBuffers returns flatbuffers::Optional
            lines.append(f"      auto {fb_field_name}_opt = fb->{fb_field_name}();")
            lines.append(f"      if ({fb_field_name}_opt.has_value()) {{")
            lines.append(f"        node.{fld.name} = {fb_field_name}_opt.value();")
            lines.append("      }")
        elif kind == "str":
            lines.append(
                f'      node.{fld.name} = fb->{fb_field_name}() ? fb->{fb_field_name}()->str() : "";'
            )
        elif kind == "optional_str":
            lines.append(f"      if (fb->{fb_field_name}()) {{")
            lines.append(f"        node.{fld.name} = fb->{fb_field_name}()->str();")
            lines.append("      }")
        elif kind == "dtype":
            lines.append(
                f"      node.{fld.name} = convert_dtype(fb->{fb_field_name}());"
            )
        elif kind == "optional_dtype":
            # Optional scalar with = null default - FlatBuffers returns flatbuffers::Optional
            lines.append(f"      auto {fb_field_name}_opt = fb->{fb_field_name}();")
            lines.append(f"      if ({fb_field_name}_opt.has_value()) {{")
            lines.append(
                f"        node.{fld.name} = convert_dtype({fb_field_name}_opt.value());"
            )
            lines.append("      }")
        elif kind == "int_or_vid":
            lines.append(
                f"      node.{fld.name} = convert_int_or_vid(fb->{fb_field_name}());"
            )
        elif kind == "int_vector" or kind == "list_int":
            lines.append(f"      node.{fld.name} = to_vector(fb->{fb_field_name}());")
        elif kind == "list_int_or_vid":
            lines.append(f"      if (fb->{fb_field_name}()) {{")
            lines.append(
                f"        for (size_t i = 0; i < fb->{fb_field_name}()->size(); ++i) {{"
            )
            lines.append(
                f"          node.{fld.name}.push_back(convert_int_or_vid(fb->{fb_field_name}()->Get(i)));"
            )
            lines.append("        }")
            lines.append("      }")

    lines.extend(
        [
            f"      instr.op = OpCode::{op_code};",
            "      instr.node = std::move(node);",
            "      break;",
            "    }",
            "",
        ]
    )

    return lines


# =============================================================================
# FlatBuffer compilation
# =============================================================================


def run_flatc(flatc_path: str = "flatc") -> bool:
    """Run flatc to generate Python and C++ bindings."""
    print(f"Running flatc on {SCHEMA_FBS}...")

    # Create output directories
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    success = True

    # Generate Python bindings
    cmd_py = [
        flatc_path,
        "--python",
        "-o",
        str(GENERATED_DIR),
        str(SCHEMA_FBS),
    ]
    try:
        result = subprocess.run(cmd_py, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"flatc (Python) failed: {result.stderr}")
            success = False
        else:
            print(f"Generated FlatBuffer Python bindings in {GENERATED_DIR}")
    except FileNotFoundError:
        print(f"flatc not found at '{flatc_path}'. Skipping FlatBuffer generation.")
        success = False

    # Generate C++ bindings
    cmd_cpp = [
        flatc_path,
        "--cpp",
        "-o",
        str(RUNTIME_DIR),
        str(SCHEMA_FBS),
    ]
    try:
        result = subprocess.run(cmd_cpp, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"flatc (C++) failed: {result.stderr}")
            success = False
        else:
            print(f"Generated FlatBuffer C++ bindings in {RUNTIME_DIR}")
    except FileNotFoundError:
        success = False

    return success


# =============================================================================
# Inspector Generation
# =============================================================================


def _get_inspector_field_kind(fld: FBSField, table: FBSTable) -> str:
    """Determine the inspector field kind for a field.

    Returns one of: 'tid', 'vid', 'int_or_vid', 'float_or_vid', 'int_list',
    'int_or_vid_list', 'scalar', 'string'
    """
    t = fld.type_str

    if t == "Tid":
        return "tid"
    if t == "Vid":
        return "vid"
    if t == "IntOrVid":
        return "int_or_vid"
    if t == "FloatOrVid":
        return "float_or_vid"
    if t == "[int32]" or t == "[int64]":
        return "int_list"
    if t == "[IntOrVid]":
        return "int_or_vid_list"
    if t == "string":
        return "string"
    # Everything else (int, float, bool, enum) is a scalar
    return "scalar"


def generate_inspector(schema: "Schema") -> str:  # noqa: F821
    """Generate the inspector field mappings file."""
    lines = [
        "#",
        "# Copyright (c) Meta Platforms, Inc. and affiliates.",
        "# All rights reserved.",
        "#",
        "# This source code is licensed under the BSD-style license found in the",
        "# LICENSE file in the root directory of this source tree.",
        "#",
        "# " + "=" * 76,
        "# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        "# " + "=" * 76,
        "#",
        "# This file was generated from schema.fbs by the MLX delegate code generator.",
        "#",
        "# Source:    backends/apple/mlx/serialization/schema.fbs",
        "# Generator: backends/apple/mlx/serialization/generate.py",
        "#",
        "# To regenerate, run from the executorch root:",
        "#     python backends/apple/mlx/serialization/generate.py",
        "#",
        "# " + "=" * 76,
        "",
        '"""',
        "Auto-generated inspector field mappings for MLX delegate.",
        "",
        "This module provides field metadata for each op node type, enabling",
        "the pte_inspector to parse FlatBuffer op nodes without manually",
        "maintaining field mappings.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any, Callable, Dict, List, Optional, Tuple",
        "",
        "",
        "# Field kinds and their extractors",
        "# Each field is a tuple of (display_name, accessor_name, kind)",
        "# where kind is one of: 'tid', 'vid', 'int_or_vid', 'float_or_vid',",
        "# 'int_list', 'int_or_vid_list', 'scalar', 'string'",
        "",
        "FieldSpec = Tuple[str, str, str]  # (display_name, accessor_name, kind)",
        "",
        "",
        "# Mapping from op node name to list of field specs",
        "OP_NODE_FIELDS: Dict[str, List[FieldSpec]] = {",
    ]

    op_nodes = schema.get_op_nodes()

    for table in op_nodes:
        lines.append(f'    "{table.name}": [')
        for fld in table.fields:
            # Skip fields ending in _is_set (legacy pattern)
            if fld.name.endswith("_is_set"):
                continue

            kind = _get_inspector_field_kind(fld, table)
            # Convert snake_case to PascalCase for accessor
            accessor = _to_pascal_case(fld.name)
            lines.append(f'        ("{fld.name}", "{accessor}", "{kind}"),')
        lines.append("    ],")

    lines.append("}")
    lines.append("")
    lines.append("")

    # Add the list of op node names for import generation
    lines.append("# List of all op node names (for dynamic imports)")
    lines.append("OP_NODE_NAMES: List[str] = [")
    for table in op_nodes:
        lines.append(f'    "{table.name}",')
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Generate MLX delegate code from schema.fbs"
    )
    parser.add_argument(
        "--flatc",
        default="flatc",
        help="Path to flatc compiler",
    )
    parser.add_argument(
        "--skip-flatc",
        action="store_true",
        help="Skip running flatc (use existing generated files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    args = parser.parse_args()

    print(f"Parsing {SCHEMA_FBS}...")
    schema = parse_fbs(SCHEMA_FBS)
    print(
        f"  Found {len(schema.enums)} enums, {len(schema.structs)} structs, "
        f"{len(schema.tables)} tables, {len(schema.unions)} unions"
    )
    print(f"  Op nodes: {len(schema.get_op_nodes())}")

    # Run flatc
    if not args.skip_flatc:
        run_flatc(args.flatc)

    # Generate Python schema
    print(f"Generating {GENERATED_SCHEMA_PY}...")
    py_schema = generate_python_schema(schema)
    if args.dry_run:
        print("--- mlx_graph_schema.py (first 50 lines) ---")
        print("\n".join(py_schema.split("\n")[:50]))
    else:
        with open(GENERATED_SCHEMA_PY, "w") as f:
            f.write(py_schema)

    # Generate Python serializers
    print(f"Generating {GENERATED_SERIALIZERS}...")
    py_serializers = generate_python_serializers(schema)
    if args.dry_run:
        print("--- _generated_serializers.py (first 50 lines) ---")
        print("\n".join(py_serializers.split("\n")[:50]))
    else:
        with open(GENERATED_SERIALIZERS, "w") as f:
            f.write(py_serializers)

    # Generate C++ header
    print(f"Generating {LOADER_H}...")
    cpp_h = generate_cpp_loader_h(schema)
    if args.dry_run:
        print("--- MLXLoader.h (first 50 lines) ---")
        print("\n".join(cpp_h.split("\n")[:50]))
    else:
        with open(LOADER_H, "w") as f:
            f.write(cpp_h)

    # Generate C++ implementation
    print(f"Generating {LOADER_CPP}...")
    cpp_cpp = generate_cpp_loader_cpp(schema)
    if args.dry_run:
        print("--- MLXLoader.cpp (first 50 lines) ---")
        print("\n".join(cpp_cpp.split("\n")[:50]))
    else:
        with open(LOADER_CPP, "w") as f:
            f.write(cpp_cpp)

    # Create __init__.py for _generated package that re-exports from mlx_delegate
    init_file = GENERATED_DIR / "__init__.py"
    if not args.dry_run:
        init_file.parent.mkdir(parents=True, exist_ok=True)

        # Get all the exports from mlx_delegate (tables, enums, structs, and unions)
        exports = []
        for table in schema.tables:
            exports.append(table.name)
        for enum in schema.enums:
            exports.append(enum.name)
        for struct in schema.structs:
            exports.append(struct.name)
        for union in schema.unions:
            exports.append(union.name)

        # Create __init__.py with re-exports
        init_content = """# Auto-generated FlatBuffer bindings
# Re-exports from mlx_delegate namespace for convenient imports

"""
        # Add imports from mlx_delegate
        for export in sorted(exports):
            init_content += f"from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.{export} import {export}\n"

        init_content += f"\n__all__ = {sorted(exports)!r}\n"
        init_file.write_text(init_content)

    # Generate inspector field mappings
    print(f"Generating {GENERATED_INSPECTOR}...")
    inspector_py = generate_inspector(schema)
    if args.dry_run:
        print("--- _generated_inspector.py (first 50 lines) ---")
        print("\n".join(inspector_py.split("\n")[:50]))
    else:
        with open(GENERATED_INSPECTOR, "w") as f:
            f.write(inspector_py)

    print("Done!")
    print("")
    print("Generated files:")
    print(f"  - {GENERATED_SCHEMA_PY}")
    print(f"  - {GENERATED_SERIALIZERS}")
    print(f"  - {GENERATED_INSPECTOR}")
    print(f"  - {LOADER_H}")
    print(f"  - {LOADER_CPP}")
    if not args.skip_flatc:
        print(f"  - {GENERATED_DIR}/ (FlatBuffer Python bindings)")
        print(f"  - {RUNTIME_DIR}/schema_generated.h (FlatBuffer C++ bindings)")


if __name__ == "__main__":
    main()
