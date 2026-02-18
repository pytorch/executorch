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
LOADER_H_TMPL = SCRIPT_DIR / "MLXLoader.h.tmpl"
LOADER_CPP_TMPL = SCRIPT_DIR / "MLXLoader.cpp.tmpl"
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
    type_str: str
    required: bool
    default: Optional[str]


# =============================================================================
# Shared type constants
# =============================================================================

# FBS integer types (signed and unsigned)
FBS_INTEGER_TYPES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)

# FBS float types
FBS_FLOAT_TYPES = frozenset({"float", "double"})

# All FBS primitive scalar types (numbers + bool)
FBS_SCALAR_TYPES = FBS_INTEGER_TYPES | FBS_FLOAT_TYPES | frozenset({"bool"})

# Compound "or" types that wrap a literal + Vid
FBS_COMPOUND_TYPES = frozenset({"IntOrVid", "FloatOrVid", "TidOrVid"})

# Python type mapping for FBS primitives
FBS_TO_PYTHON = {
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "float": "float",
    "double": "float",
    "bool": "bool",
    "string": "str",
    "byte": "int",
}

# C++ type mapping for FBS primitives
FBS_TO_CPP = {
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float": "float",
    "double": "double",
    "bool": "bool",
    "string": "std::string",
    "byte": "uint8_t",
    "Tid": "Tid",
    "Vid": "Vid",
    "IntOrVid": "std::variant<int64_t, Vid>",
    "FloatOrVid": "std::variant<double, Vid>",
}


def _section_header(comment: str, title: str) -> List[str]:
    """Generate a section-header banner for generated output."""
    sep = f"{comment} {'=' * 76}"
    return [sep, f"{comment} {title}", sep, ""]


def _file_header(comment: str, description: str = "") -> List[str]:
    """Generate a standard auto-generated file header.

    Args:
        comment: Comment prefix, e.g. '#' for Python or '//' for C++.
        description: Optional description appended after the banner.
    """
    sep = f"{comment} {'=' * 76}"
    lines = [
        f"{comment}",
        f"{comment} Copyright (c) Meta Platforms, Inc. and affiliates.",
        f"{comment} All rights reserved.",
        f"{comment}",
        f"{comment} This source code is licensed under the BSD-style license found in the",
        f"{comment} LICENSE file in the root directory of this source tree.",
        f"{comment}",
        sep,
        f"{comment} AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        sep,
        f"{comment}",
        f"{comment} This file was generated from schema.fbs by the MLX delegate code generator.",
        f"{comment}",
        f"{comment} Source:    backends/apple/mlx/serialization/schema.fbs",
        f"{comment} Generator: backends/apple/mlx/serialization/generate.py",
        f"{comment}",
        f"{comment} To regenerate, run from the executorch root:",
        f"{comment}     python backends/apple/mlx/serialization/generate.py",
        f"{comment}",
        sep,
    ]
    if description:
        lines.append(f"{comment}")
        lines.append(f"{comment} {description}")
    return lines


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


# Config for compound type factory methods.
# Maps compound type name -> (primary_field_name, primary_python_type, description)
_COMPOUND_TYPE_CONFIG = {
    "IntOrVid": ("literal", "int", "a literal integer"),
    "FloatOrVid": ("literal", "float", "a literal float"),
    "TidOrVid": ("tid", "Tid", "a tensor reference"),
}


def _generate_compound_type(table: FBSTable) -> List[str]:
    """Generate a Python dataclass for a compound type (IntOrVid, etc.) from schema."""
    name = table.name
    config = _COMPOUND_TYPE_CONFIG.get(name)
    if not config:
        raise ValueError(f"No compound type config for '{name}'")

    primary_field, primary_py_type, primary_desc = config

    # Build the docstring from the schema structure
    lines = [
        "@dataclass",
        f"class {name}:",
    ]

    # Docstring: describe the two alternatives
    lines.append(
        f'    """Represents either {primary_desc} or a runtime Vid reference."""'
    )

    # Dataclass fields from the parsed schema
    for fld in table.fields:
        if fld.default == "false":
            default = "False"
        elif fld.default == "true":
            default = "True"
        elif fld.type_str in ("Tid", "Vid"):
            default = "None"
        elif fld.default is not None:
            default = fld.default
        elif fld.type_str in FBS_INTEGER_TYPES:
            default = "0"
        elif fld.type_str in FBS_FLOAT_TYPES:
            default = "0.0"
        else:
            default = "None"
        truly_required = default != "None"
        py_type = _fbs_type_to_python(fld.type_str, truly_required)
        lines.append(f"    {fld.name}: {py_type} = {default}")

    # Factory: from_primary (e.g. from_literal, from_tid)
    lines.append("")
    lines.append("    @classmethod")
    lines.append(
        f'    def from_{primary_field}(cls, value: {primary_py_type}) -> "{name}":'
    )
    lines.append(f'        """Create a {name} from {primary_desc}."""')
    lines.append(f"        return cls({primary_field}=value, is_vid=False)")

    # Factory: from_vid
    lines.append("")
    lines.append("    @classmethod")
    lines.append(f'    def from_vid(cls, vid: Vid) -> "{name}":')
    lines.append(f'        """Create a {name} from a Vid reference."""')
    lines.append("        return cls(vid=vid, is_vid=True)")

    lines.append("")
    return lines


def _generate_dataclass(table: FBSTable) -> List[str]:
    """Generate a Python @dataclass from a parsed FBS table.

    Handles field ordering (required/defaulted before optional), skips
    _is_set sentinel fields, and emits proper type annotations with defaults.
    """
    lines = ["@dataclass", f"class {table.name}:"]
    fields = [f for f in table.fields if not f.name.endswith("_is_set")]
    if not fields:
        lines.append("    pass")
    else:
        required_fields = [f for f in fields if f.required or f.default is not None]
        optional_fields = [f for f in fields if not f.required and f.default is None]

        for fld in required_fields:
            py_type = _fbs_type_to_python(fld.type_str, True)
            default = _fbs_default_to_python(fld.default, fld.type_str)
            if default is not None:
                lines.append(f"    {fld.name}: {py_type} = {default}")
            else:
                lines.append(f"    {fld.name}: {py_type}")

        for fld in optional_fields:
            py_type = _fbs_type_to_python(fld.type_str, fld.required)
            lines.append(f"    {fld.name}: {py_type} = None")

    lines.extend(["", ""])
    return lines


# =============================================================================
# Python dataclass generation
# =============================================================================


def generate_python_schema(schema: FBSSchema) -> str:  # noqa: C901
    """Generate mlx_graph_schema.py from parsed FBS."""
    lines = _file_header("#")
    lines.extend(
        [
            "",
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass, field",
            "from enum import IntEnum",
            "from typing import List, Optional, Union",
            "",
            "",
            *_section_header("#", "Enums"),
        ]
    )

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

    lines.extend(_section_header("#", "Core types"))

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

    # Generate compound types (IntOrVid, FloatOrVid, TidOrVid) from schema
    for type_name in sorted(FBS_COMPOUND_TYPES):
        table = next((t for t in schema.tables if t.name == type_name), None)
        if table:
            lines.extend(_generate_compound_type(table))
            lines.append("")

    # Generate SlotVariant, NamedSlot, TensorMeta (but not Instruction/MLXGraph yet - they reference OpNode)
    other_tables = ["SlotVariant", "NamedSlot", "TensorMeta"]
    for table_name in other_tables:
        table = next((t for t in schema.tables if t.name == table_name), None)
        if table:
            lines.extend(_generate_dataclass(table))

    lines.extend(_section_header("#", "Op nodes"))

    # Generate op node dataclasses
    op_nodes = schema.get_op_nodes()
    for table in op_nodes:
        lines.extend(_generate_dataclass(table))

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
            *_section_header("#", "Container types (reference OpNodeUnion)"),
            "@dataclass",
            "class Instruction:",
            "    op: OpNodeUnion",
            "",
            "",
            "@dataclass",
            "class InstructionChain:",
            "    instructions: List[Instruction]",
            "",
            "",
            "@dataclass",
            "class MLXGraph:",
            "    instruction_chains: List[InstructionChain]",
            "    version: Optional[str] = None",
            "    num_constant_tensors: int = 0",
            "    num_input_tensors: int = 0",
            "    num_output_tensors: int = 0",
            "    num_mutable_buffer_tensors: int = 0",
            "    num_temp_tensors: int = 0",
            "    num_values: int = 0",
            "    main_chain_idx: int = 0",
            "    init_chain_idx: int = -1",
            "    input_map: Optional[List[SlotVariant]] = None",
            "    output_map: Optional[List[SlotVariant]] = None",
            "    mutable_buffer_map: Optional[List[SlotVariant]] = None",
            "    named_slots: Optional[List[NamedSlot]] = None",
            "    tensor_meta: Optional[List[TensorMeta]] = None",
            "",
        ]
    )

    return "\n".join(lines)


def _fbs_type_to_python(fbs_type: str, required: bool) -> str:
    """Convert FBS type to Python type annotation.

    When required=False, the result is wrapped in Optional[…] for all types
    (scalars, lists, and reference types alike).
    """
    # Handle arrays
    if fbs_type.startswith("[") and fbs_type.endswith("]"):
        inner = fbs_type[1:-1]
        inner_py = _fbs_type_to_python(inner, True)
        base = f"List[{inner_py}]"
        return base if required else f"Optional[{base}]"

    py_type = FBS_TO_PYTHON.get(fbs_type, fbs_type)

    if not required:
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
    op_union = next((u for u in schema.unions if u.name == "OpNode"), None)

    header = _file_header(
        "#",
        "This file contains auto-generated serializer methods for all op types.",
    )

    # Imports and module-level code
    op_imports = ",\n".join(f"    {t.name}" for t in op_nodes)
    lines = [
        *header,
        "",
        "from __future__ import annotations",
        "",
        "from typing import List, Tuple, Dict",
        "",
        "import flatbuffers",
        "",
    ]

    # Generate op type names dict from union order
    lines.append(
        "# FlatBuffer union indices: 0 = NONE, then 1-indexed from union order"
    )
    lines.append("MLX_OP_TYPE_NAMES = {")
    lines.append('    0: "NONE",')
    if op_union:
        for i, type_name in enumerate(op_union.types, start=1):
            lines.append(f'    {i}: "{type_name}",')
    lines.append("}")
    lines.append("")

    lines.extend(
        [
            "from executorch.backends.apple.mlx.serialization.mlx_graph_schema import (",
            f"{op_imports},",
            "    IntOrVid,",
            "    FloatOrVid,",
            "    TidOrVid,",
            "    Tid,",
            "    Vid,",
            ")",
            "",
            "",
            "def _build_int_vector(builder: flatbuffers.Builder, vec: List[int]) -> int:",
            '    """Build a vector of int32."""',
            "    builder.StartVector(4, len(vec), 4)",
            "    for v in reversed(vec):",
            "        builder.PrependInt32(v)",
            "    return builder.EndVector()",
            "",
            "",
            "class GeneratedOpBuilders:",
            '    """Mixin class with auto-generated op builder methods."""',
            "",
            "    def _build_int_or_vid(self, builder: flatbuffers.Builder, iov: IntOrVid) -> int:",
            '        """Build an IntOrVid table."""',
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import IntOrVid as FBIntOrVidModule",
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid",
            "",
            "        FBIntOrVidModule.Start(builder)",
            "        FBIntOrVidModule.AddLiteral(builder, iov.literal)",
            "        FBIntOrVidModule.AddIsVid(builder, iov.is_vid)",
            "        if iov.vid is not None:",
            "            # Vid is an inline struct - must be added last for proper FlatBuffer layout",
            "            FBIntOrVidModule.AddVid(builder, CreateVid(builder, iov.vid.idx))",
            "        return FBIntOrVidModule.End(builder)",
            "",
            "    def _build_float_or_vid(self, builder: flatbuffers.Builder, fov: FloatOrVid) -> int:",
            '        """Build a FloatOrVid table."""',
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import FloatOrVid as FBFloatOrVidModule",
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid",
            "",
            "        FBFloatOrVidModule.Start(builder)",
            "        FBFloatOrVidModule.AddLiteral(builder, fov.literal)",
            "        FBFloatOrVidModule.AddIsVid(builder, fov.is_vid)",
            "        if fov.vid is not None:",
            "            FBFloatOrVidModule.AddVid(builder, CreateVid(builder, fov.vid.idx))",
            "        return FBFloatOrVidModule.End(builder)",
            "",
            "    def _build_tid_or_vid(self, builder: flatbuffers.Builder, tov: TidOrVid) -> int:",
            '        """Build a TidOrVid table."""',
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate import TidOrVid as FBTidOrVidModule",
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid",
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Vid import CreateVid",
            "",
            "        FBTidOrVidModule.Start(builder)",
            "        FBTidOrVidModule.AddIsVid(builder, tov.is_vid)",
            "        if tov.tid is not None:",
            "            FBTidOrVidModule.AddTid(builder, CreateTid(builder, tov.tid.idx))",
            "        if tov.vid is not None:",
            "            FBTidOrVidModule.AddVid(builder, CreateVid(builder, tov.vid.idx))",
            "        return FBTidOrVidModule.End(builder)",
            "",
            "    def _build_int_or_vid_vector(",
            "        self, builder: flatbuffers.Builder, vec: List[IntOrVid]",
            "    ) -> int:",
            '        """Build a vector of IntOrVid tables."""',
            "        offsets = []",
            "        for iov in vec:",
            "            offsets.append(self._build_int_or_vid(builder, iov))",
            "        builder.StartVector(4, len(offsets), 4)",
            "        for off in reversed(offsets):",
            "            builder.PrependUOffsetTRelative(off)",
            "        return builder.EndVector()",
            "",
            "    def _build_tid_vector(",
            "        self, builder: flatbuffers.Builder, vec: List[Tid]",
            "    ) -> int:",
            '        """Build a vector of Tid structs."""',
            "        from executorch.backends.apple.mlx.serialization._generated.mlx_delegate.Tid import CreateTid",
            "",
            "        # For vectors of structs, we need to build the vector differently",
            "        # Each Tid struct is 4 bytes (uint32), so we manually write them",
            "        builder.StartVector(4, len(vec), 4)",
            "        for tid in reversed(vec):",
            "            builder.Prep(4, 0)  # Align for struct",
            "            builder.PrependUint32(tid.idx)",
            "        return builder.EndVector()",
            "",
        ]
    )

    # Generate builder methods for each op
    for table in op_nodes:
        lines.append(_generate_op_builder_method(table))

    return "\n".join(lines)


def _generate_op_builder_method(table: FBSTable) -> str:
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
        pb = _emit_py_prebuild(kind, fld)
        if pb:
            prebuild_lines.extend(pb)

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
        add_lines = _emit_py_add(kind, fld, fb_module_name, fb_field_name)
        if add_lines is None:
            raise ValueError(
                f"Unhandled field kind '{kind}' for field '{fld.name}' in table '{table.name}'. "
                f"Add a handler in _emit_py_add()."
            )
        lines.extend(add_lines)

    # End the FlatBuffer table and return offset + union type
    lines.append(f"        offset = {fb_module_name}.End(builder)")
    lines.append(f"        return offset, FBOpNodeModule.OpNode.{class_name}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Python builder emitters — data-driven per field kind
# ---------------------------------------------------------------------------

# Prebuild emitters: return list of lines or None if no prebuild needed.
# These build offsets/vectors that must be created before FlatBuffer Start().

_PY_PREBUILD_VECTOR = {
    "list_int": "_build_int_vector(builder, op.{name})",
    "list_int_or_vid": "self._build_int_or_vid_vector(builder, op.{name})",
    "list_tid": "self._build_tid_vector(builder, op.{name})",
}

_PY_PREBUILD_OFFSET = {
    "str": "builder.CreateString(op.{name})",
    "int_or_vid": "self._build_int_or_vid(builder, op.{name})",
    "float_or_vid": "self._build_float_or_vid(builder, op.{name})",
    "tid_or_vid": "self._build_tid_or_vid(builder, op.{name})",
    "optional_str": "builder.CreateString(op.{name}) if op.{name} is not None else None",
}


def _emit_py_prebuild(kind: str, fld: FBSField) -> List[str]:
    """Emit prebuild lines for a field kind, or empty list if none needed."""
    n = fld.name
    if kind in _PY_PREBUILD_VECTOR:
        expr = _PY_PREBUILD_VECTOR[kind].format(name=n)
        if fld.required:
            return [f"        {n}_vec = {expr}"]
        else:
            return [f"        {n}_vec = {expr} if op.{n} is not None else None"]
    if kind in _PY_PREBUILD_OFFSET:
        suffix = "_off"
        expr = _PY_PREBUILD_OFFSET[kind].format(name=n)
        return [f"        {n}{suffix} = {expr}"]
    return []


# Maps struct kinds to their Python Create function name
_PY_STRUCT_CREATOR = {"tid": "CreateTid", "vid": "CreateVid"}


def _emit_py_add(
    kind: str, fld: FBSField, mod: str, fb_name: str
) -> "List[str] | None":
    """Emit Add lines for a field kind, or None if kind is unrecognized."""
    n = fld.name
    add = f"{mod}.Add{fb_name}"

    # Required struct via inline Create call
    if kind in _PY_STRUCT_CREATOR:
        creator = _PY_STRUCT_CREATOR[kind]
        return [f"        {add}(builder, {creator}(builder, op.{n}.idx))"]
    # Scalars (direct value)
    if kind in ("int", "float", "bool"):
        return [f"        {add}(builder, op.{n})"]
    # Pre-built offsets (string, compound types)
    if kind in ("str", "int_or_vid", "float_or_vid", "tid_or_vid"):
        return [f"        {add}(builder, {n}_off)"]
    # Pre-built vectors (required vs optional)
    if kind in ("list_int", "list_int_or_vid", "list_tid"):
        if fld.required:
            return [f"        {add}(builder, {n}_vec)"]
        return [
            f"        if {n}_vec is not None:",
            f"            {add}(builder, {n}_vec)",
        ]
    # Optional struct via inline Create call
    if kind in ("optional_tid", "optional_vid"):
        creator = _PY_STRUCT_CREATOR[kind.removeprefix("optional_")]
        return [
            f"        if op.{n} is not None:",
            f"            {add}(builder, {creator}(builder, op.{n}.idx))",
        ]
    # Optional scalars
    if kind in ("optional_float", "optional_int"):
        return [
            f"        if op.{n} is not None:",
            f"            {add}(builder, op.{n})",
        ]
    # Optional string offset
    if kind == "optional_str":
        return [
            f"        if {n}_off is not None:",
            f"            {add}(builder, {n}_off)",
        ]
    return None


def _get_field_kind(fld: FBSField, table: FBSTable) -> str:  # noqa: C901
    """Classify a field into a canonical kind string.

    This is the single source of truth for field classification, used by all
    generators (Python builder, C++ loader, and inspector via _INSPECTOR_KIND_MAP).
    """
    t = fld.type_str

    # Handle arrays
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1]
        if inner in FBS_INTEGER_TYPES:
            return "list_int"
        if inner == "IntOrVid":
            return "list_int_or_vid"
        if inner == "Tid":
            return "list_tid"
        raise ValueError(
            f"Unrecognized array element type '{inner}' for field '{fld.name}' in table '{table.name}'. "
            f"Add a handler in _get_field_kind()."
        )

    # Handle basic types
    if t == "Tid":
        return "optional_tid" if not fld.required else "tid"
    if t == "Vid":
        return "optional_vid" if not fld.required else "vid"
    if t == "IntOrVid":
        return "int_or_vid"
    if t == "FloatOrVid":
        return "float_or_vid"
    if t == "TidOrVid":
        return "tid_or_vid"
    if t in FBS_INTEGER_TYPES:
        if fld.default == "null":
            return "optional_int"
        return "int"
    if t in FBS_FLOAT_TYPES:
        # Check if this is optional (has = null default)
        if fld.default == "null":
            return "optional_float"
        return "float"
    if t == "bool":
        return "bool"
    if t == "string":
        return "optional_str" if not fld.required else "str"

    raise ValueError(
        f"Unrecognized field type '{t}' for field '{fld.name}' in table '{table.name}'. "
        f"Add a handler in _get_field_kind()."
    )


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
    """Generate MLXLoader.h from parsed FBS using template."""
    op_nodes = schema.get_op_nodes()

    # --- Dynamic part 1: op node structs ---
    struct_lines = []
    for table in op_nodes:
        struct_lines.append(f"struct {table.name} {{")
        if not table.fields:
            struct_lines.append("};")
        else:
            for fld in table.fields:
                if fld.name.endswith("_is_set"):
                    continue
                cpp_type = _fbs_type_to_cpp(fld.type_str, fld.required, table, fld)
                struct_lines.append(f"  {cpp_type} {fld.name};")
            struct_lines.append("};")
        struct_lines.append("")

    # --- Dynamic part 2: OpCode enum values ---
    enum_lines = []
    for table in op_nodes:
        enum_lines.append(f"  {_table_name_to_opcode(table.name)},")

    # --- Dynamic part 3: op_name() switch cases ---
    name_lines = []
    for table in op_nodes:
        op_code = _table_name_to_opcode(table.name)
        name_lines.append(f"    case OpCode::{op_code}:")
        name_lines.append(f'      return "{op_code}";')

    # --- Dynamic part 4: NodeVariant type list ---
    variant_lines = []
    for i, table in enumerate(op_nodes):
        comma = "," if i < len(op_nodes) - 1 else ""
        variant_lines.append(f"    {table.name}{comma}")

    # Read template and fill placeholders
    header = "\n".join(_file_header("//")) + "\n//\n"
    tmpl = LOADER_H_TMPL.read_text()
    result = tmpl.replace("{{OP_NODE_STRUCTS}}", "\n".join(struct_lines))
    result = result.replace("{{OPCODE_ENUM_VALUES}}", "\n".join(enum_lines))
    result = result.replace("{{OP_NAME_CASES}}", "\n".join(name_lines))
    result = result.replace("{{NODE_VARIANT_TYPES}}", "\n".join(variant_lines))
    return header + result


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

    Note: Most scalar types (float, int, etc.) are never optional in C++.
    The Python serialization layer is responsible for ensuring scalar fields
    have values (using defaults if user doesn't provide them).
    Reference types (Tid, Vid) and DTypeId with '= null' default can be optional.
    """
    # Handle arrays
    if fbs_type.startswith("[") and fbs_type.endswith("]"):
        inner = fbs_type[1:-1]
        inner_cpp = _fbs_type_to_cpp(inner, True)
        return f"std::vector<{inner_cpp}>"

    cpp_type = FBS_TO_CPP.get(fbs_type, fbs_type)

    # Handle optional types
    if not required:
        if fbs_type == "Tid":
            return "std::optional<Tid>"
        if fbs_type == "Vid":
            return "std::optional<Vid>"
        if fld is not None and fld.default == "null" and fbs_type in FBS_TO_CPP:
            return f"std::optional<{cpp_type}>"

    return cpp_type


_OPCODE_OVERRIDES = {
    "ARange": "ARANGE",
    "AsType": "ASTYPE",
    "Conv1D": "CONV1D",
    "Conv2D": "CONV2D",
    "Conv3D": "CONV3D",
}


def _table_name_to_opcode(name: str) -> str:
    """Convert table name like 'LinearNode' to opcode like 'LINEAR'.

    Uses regex-based camelCase → UPPER_SNAKE_CASE conversion with a small
    override dict for names whose conventional opcode doesn't follow the
    normal camelCase splitting rules (e.g. Conv1D → CONV1D, not CONV1_D).
    """
    name = name.removesuffix("Node")
    if name in _OPCODE_OVERRIDES:
        return _OPCODE_OVERRIDES[name]
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.upper()


def generate_cpp_loader_cpp(schema: FBSSchema) -> str:
    """Generate MLXLoader.cpp from parsed FBS using template."""
    op_nodes = schema.get_op_nodes()

    # --- Dynamic part: switch cases for load_instruction ---
    case_lines = []
    for table in op_nodes:
        case_lines.extend(_generate_loader_case(table))

    # Read template and fill placeholders
    header = "\n".join(_file_header("//")) + "\n"
    tmpl = LOADER_CPP_TMPL.read_text()
    result = tmpl.replace("{{LOAD_INSTRUCTION_CASES}}", "\n".join(case_lines))
    return header + result


def _generate_loader_case(table: FBSTable) -> List[str]:
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

        fb_field_name = fld.name
        kind = _get_field_kind(fld, table)
        load_lines = _emit_cpp_load(kind, fld.name, fb_field_name)
        if load_lines is None:
            raise ValueError(
                f"Unhandled field kind '{kind}' for field '{fld.name}' in table '{table.name}'. "
                f"Add a handler in _emit_cpp_load()."
            )
        lines.extend(load_lines)

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


# ---------------------------------------------------------------------------
# C++ loader emitters — data-driven per field kind
# ---------------------------------------------------------------------------


# Maps kinds to their C++ converter function name
_CPP_CONVERTER = {
    "tid": "convert_tid",
    "vid": "convert_vid",
    "int_or_vid": "convert_int_or_vid",
    "float_or_vid": "convert_float_or_vid",
    "tid_or_vid": "convert_tid_or_vid",
}


def _emit_cpp_load(kind: str, name: str, fb_name: str) -> "List[str] | None":
    """Emit C++ load lines for a field kind, or None if kind is unrecognized."""
    # Required struct / compound via converter
    if kind in _CPP_CONVERTER:
        conv = _CPP_CONVERTER[kind]
        return [f"      node.{name} = {conv}(fb->{fb_name}());"]
    # Scalars (direct value)
    if kind in ("int", "float", "bool"):
        return [f"      node.{name} = fb->{fb_name}();"]
    # Required string
    if kind == "str":
        return [f'      node.{name} = fb->{fb_name}() ? fb->{fb_name}()->str() : "";']
    # Optional struct / compound via guarded converter
    base_kind = kind.removeprefix("optional_")
    if kind.startswith("optional_") and base_kind in _CPP_CONVERTER:
        conv = _CPP_CONVERTER[base_kind]
        return [
            f"      if (fb->{fb_name}()) {{",
            f"        node.{name} = {conv}(fb->{fb_name}());",
            "      }",
        ]
    # Optional scalar (FlatBuffers returns flatbuffers::Optional)
    if kind in ("optional_float", "optional_int"):
        return [
            f"      auto {fb_name}_opt = fb->{fb_name}();",
            f"      if ({fb_name}_opt.has_value()) {{",
            f"        node.{name} = {fb_name}_opt.value();",
            "      }",
        ]
    # Optional string
    if kind == "optional_str":
        return [
            f"      if (fb->{fb_name}()) {{",
            f"        node.{name} = fb->{fb_name}()->str();",
            "      }",
        ]
    # Integer/bool vector via to_vector
    if kind == "list_int":
        return [f"      node.{name} = to_vector(fb->{fb_name}());"]
    # Int-or-vid vector (indexed access)
    if kind == "list_int_or_vid":
        return [
            f"      if (fb->{fb_name}()) {{",
            f"        for (size_t i = 0; i < fb->{fb_name}()->size(); ++i) {{",
            f"          node.{name}.push_back(convert_int_or_vid(fb->{fb_name}()->Get(i)));",
            "        }",
            "      }",
        ]
    # Tid vector (range-based iteration)
    if kind == "list_tid":
        return [
            f"      if (fb->{fb_name}()) {{",
            f"        for (auto fb_tid : *fb->{fb_name}()) {{",
            f"          node.{name}.push_back(convert_tid(fb_tid));",
            "        }",
            "      }",
        ]
    return None


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


# Mapping from fine-grained field kinds (from _get_field_kind) to inspector
# display kinds.  The inspector uses coarser categories: optional/required
# distinctions collapse, and int/float/bool all map to "scalar".
_INSPECTOR_KIND_MAP = {
    "tid": "tid",
    "optional_tid": "tid",
    "vid": "vid",
    "optional_vid": "vid",
    "int_or_vid": "int_or_vid",
    "float_or_vid": "float_or_vid",
    "tid_or_vid": "tid_or_vid",
    "list_int": "int_list",
    "list_int_or_vid": "int_or_vid_list",
    "list_tid": "tid_list",
    "int": "scalar",
    "optional_int": "scalar",
    "float": "scalar",
    "optional_float": "scalar",
    "bool": "scalar",
    "str": "string",
    "optional_str": "string",
}


def generate_inspector(schema: "Schema") -> str:  # noqa: F821
    """Generate the inspector field mappings file."""
    lines = _file_header("#")
    lines.extend(
        [
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
            "from typing import Dict, List, Tuple",
            "",
            "",
            "# Field kinds and their extractors",
            "# Each field is a tuple of (display_name, accessor_name, kind)",
            "# where kind is one of: 'tid', 'vid', 'int_or_vid', 'float_or_vid',",
            "# 'int_list', 'int_or_vid_list', 'tid_list', 'scalar', 'string'",
            "",
            "FieldSpec = Tuple[str, str, str]  # (display_name, accessor_name, kind)",
            "",
            "",
            "# Mapping from op node name to list of field specs",
            "OP_NODE_FIELDS: Dict[str, List[FieldSpec]] = {",
        ]
    )

    op_nodes = schema.get_op_nodes()

    for table in op_nodes:
        lines.append(f'    "{table.name}": [')
        for fld in table.fields:
            # Skip fields ending in _is_set (legacy pattern)
            if fld.name.endswith("_is_set"):
                continue

            kind = _get_field_kind(fld, table)
            inspector_kind = _INSPECTOR_KIND_MAP.get(kind)
            if inspector_kind is None:
                raise ValueError(
                    f"No inspector mapping for field kind '{kind}' "
                    f"(field '{fld.name}' in table '{table.name}'). "
                    f"Add a mapping in _INSPECTOR_KIND_MAP."
                )
            accessor = _to_pascal_case(fld.name)
            lines.append(f'        ("{fld.name}", "{accessor}", "{inspector_kind}"),')
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

    # Generate all code files
    generators = [
        (generate_python_schema, GENERATED_SCHEMA_PY, "mlx_graph_schema.py"),
        (
            generate_python_serializers,
            GENERATED_SERIALIZERS,
            "_generated_serializers.py",
        ),
        (generate_cpp_loader_h, LOADER_H, "MLXLoader.h"),
        (generate_cpp_loader_cpp, LOADER_CPP, "MLXLoader.cpp"),
        (generate_inspector, GENERATED_INSPECTOR, "_generated_inspector.py"),
    ]
    for gen_fn, output_path, label in generators:
        print(f"Generating {output_path}...")
        content = gen_fn(schema)
        if args.dry_run:
            print(f"--- {label} (first 50 lines) ---")
            print("\n".join(content.split("\n")[:50]))
        else:
            with open(output_path, "w") as f:
                f.write(content)

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
