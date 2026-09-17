#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate runtime/ops/<op>/<stem>_wgsl.h from each <stem>.wgsl.

Each header embeds the shader text unchanged as `inline constexpr const char*
k<Pascal>WGSL` plus `k<Pascal>WorkgroupSize` (parsed from @workgroup_size).

Usage:
  gen_wgsl_headers.py            # (re)write all <stem>_wgsl.h
  gen_wgsl_headers.py --check    # exit 1 if any committed header is stale

A shader is treated as a template iff a sibling <stem>.yaml spec exists; the
$-block engine (preprocess/escape/generate_variant_combinations) expands one
template + a DTYPE/VEC variant matrix into the concrete per-variant headers.

Spec parsing uses PyYAML (a declared ExecuTorch codegen dependency, mirroring
backends/vulkan/runtime/gen_vulkan_spv.py); run under the ExecuTorch dev env.
"""

import argparse
import copy
import hashlib
import io
import os
import re
import stat
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore[assignment, misc]

BACKEND_ROOT = Path(__file__).resolve().parents[1]

_SHA_RE = re.compile(r"// wgsl-sha256: ([0-9a-f]{64})")

_BSD_HEADER = """\
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */"""


########################################################################
#  WGSL template engine
#
#  A $-block transpiler (extract_leading_whitespace / escape / preprocess)
#  plus a DTYPE/VEC variant matrix (generate_variant_combinations /
#  parse_template_spec) expand one template + its YAML sidecar into the
#  per-variant WGSL headers.
########################################################################


# WGSL type-helpers injected into preprocess's exec globals so ${...} template
# expressions can spell WGSL types (f32/f16, vec4<T>); @group/@binding layout is
# written directly in the templates. Names mirror gen_vulkan_spv.py's
# buffer_scalar_type / buffer_gvec_type / accum_scalar_type.
def buffer_scalar_type(dtype: str) -> str:
    if dtype == "half":
        return "f16"
    elif dtype == "float":
        return "f32"
    return dtype


def buffer_gvec_type(dtype: str, n: int) -> str:
    if n == 1:
        return buffer_scalar_type(dtype)
    return f"vec{n}<{buffer_scalar_type(dtype)}>"


def accum_scalar_type(dtype: str) -> str:
    # The float family (incl. half) accumulates in f32 -- f16 accumulation is
    # numerically unsafe on target GPUs. Mirrors gen_vulkan_spv.py's
    # accum_scalar_type (half -> rgba16f -> "float" there is the same intent).
    if dtype in ("half", "float"):
        return "f32"
    return buffer_scalar_type(dtype)


WGSL_HELPERS: Dict[str, Any] = {
    "buffer_scalar_type": buffer_scalar_type,
    "buffer_gvec_type": buffer_gvec_type,
    "accum_scalar_type": accum_scalar_type,
}


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def extract_leading_whitespace(line: str) -> str:
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def escape(line: str) -> str:
    output_parts = []
    while "${" in line:
        start_pos = line.index("${")
        end_pos = line.index("}", start_pos + 2)
        if start_pos != 0:
            output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
        output_parts.append("str(" + line[start_pos + 2 : end_pos] + ")")
        line = line[end_pos + 1 :]
    if line:
        output_parts.append('"' + line.replace('"', '\\"') + '"')
    return " + ".join(output_parts)


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def preprocess(
    input_text: str, variables: Dict[str, Any], input_path: str = "codegen"
) -> str:
    # Normalize line endings first. Templates checked out with CRLF (common on
    # Windows) otherwise break the trailing-backslash handling below: in
    # re.MULTILINE, $ matches immediately before \n, so a CR would sit between a
    # trailing \ and the line end and defeat the r"\\$" match, leaving a lone
    # backslash that escape() turns into an unterminated Python string literal.
    input_text = input_text.replace("\r\n", "\n").replace("\r", "\n")
    # Workaround to handle source files using \ to extend mecros to a new line
    input_text = re.sub(r"\\$", r"\\\\", input_text, flags=re.MULTILINE)

    input_lines = input_text.splitlines()
    python_lines = []

    blank_lines = 0

    last_indent = ""

    # List of tuples (total_index, python_indent)
    indent_stack = [("", "")]

    # Indicates whether this is the first line inside Python
    # code block (i.e. for, while, if, elif, else)
    python_block_start = True
    for input_line in input_lines:
        if input_line == "":
            blank_lines += 1
            continue
        # Skip lint markers.
        if "LINT" in input_line:
            continue

        input_indent = extract_leading_whitespace(input_line)
        if python_block_start:
            assert input_indent.startswith(last_indent)
            extra_python_indent = input_indent[len(last_indent) :]
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((input_indent, python_indent))
            assert input_indent.startswith(indent_stack[-1][0])
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        python_block_start = False

        python_indent = indent_stack[-1][1]
        stripped_input_line = input_line.strip()
        if stripped_input_line.startswith("$") and not stripped_input_line.startswith(
            "${"
        ):
            if stripped_input_line.endswith(":"):
                python_block_start = True
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + stripped_input_line.replace("$", ""))
        else:
            assert input_line.startswith(python_indent)
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(
                python_indent
                + "print(%s, file=OUT_STREAM)"
                % escape(input_line[len(python_indent) :])
            )
        last_indent = input_indent

    while blank_lines != 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    exec_globals = dict(variables)
    output_stream = io.StringIO()
    exec_globals["OUT_STREAM"] = output_stream

    python_bytecode = compile("\n".join(python_lines), input_path, "exec")
    exec(python_bytecode, exec_globals)

    return output_stream.getvalue()


# https://gist.github.com/pypt/94d747fe5180851196eb
class UniqueKeyLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            try:
                hash(key)
            except TypeError as e:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unacceptable key ",
                    key_node.start_mark,
                ) from e
            # check for duplicate keys
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key",
                    key_node.start_mark,
                )
            value = self.construct_object(value_node, deep=deep)  # type: ignore[no-untyped-call]
            mapping[key] = value
        return mapping


def generate_variant_combinations(  # noqa: C901
    iterated_params: Dict[str, Any],
    exclude_params: Optional[Set[str]] = None,
) -> List[Any]:
    if exclude_params is None:
        exclude_params = set()
    all_iterated_params = []
    for param_name, value_list in iterated_params.items():
        if re.match(r"^combination\d*$", param_name):
            param_values = []
            param_names = value_list["parameter_names"]
            combos = value_list["combos"]
            for combo in combos:
                parameter_values = combo["parameter_values"]
                if "suffix" in combo:
                    suffix = combo["suffix"]
                else:
                    suffix = ""
                    for param_value in parameter_values:
                        if len(str(param_value)) > 0:
                            suffix += "_" + str(param_value)
                    suffix = suffix[1:]
                param_values.append((param_names, suffix, parameter_values))

            all_iterated_params.append(param_values)

        elif param_name not in exclude_params:
            param_values = []
            for value in value_list:
                if "RANGE" in value:
                    value_range = value["RANGE"]
                    suffix = value.get("SUFFIX", "")
                    if isinstance(value_range, list) and len(value_range) == 2:
                        for i in range(value_range[0], value_range[1] + 1):
                            curr_suffix = suffix + "_" + str(i) if suffix else str(i)
                            param_values.append((param_name, curr_suffix, i))
                    else:
                        raise ValueError(
                            f"{value['RANGE']} is not a valid range. Must be in format [start, end] (inclusive)."
                        )

                elif "VALUE" in value:
                    suffix = value.get("SUFFIX", value["VALUE"])
                    if value["VALUE"] in ["int", "uint"]:
                        raise ValueError(
                            f"Use int32 or uint32 instead of {value['VALUE']}"
                        )
                    param_values.append((param_name, suffix, value["VALUE"]))

                else:
                    raise KeyError(
                        "Parameter must be 'VALUE: string' or 'RANGE: [a, b]'"
                    )

            all_iterated_params.append(param_values)

    return list(product(*all_iterated_params))


def parse_template_spec(yaml_path) -> Dict[str, List[Dict[str, Any]]]:  # noqa: C901
    """Parse a <stem>.yaml variant spec into {template_name: [expanded
    per-variant param dicts]}. PyYAML with a dup-key-rejecting UniqueKeyLoader
    (mirrors gen_vulkan_spv.py)."""
    shader_template_params: Dict[str, List[Dict[str, Any]]] = {}
    with open(yaml_path) as f:
        contents = yaml.load(f, Loader=UniqueKeyLoader)
    for template_name, params_dict in contents.items():
        if template_name in shader_template_params:
            raise KeyError(f"{template_name} params file is defined twice")

        default_params = params_dict["parameter_names_with_default_values"]
        params_names = set(default_params.keys()).union({"NAME"})

        shader_template_params[template_name] = []

        default_iterated_params = params_dict.get("generate_variant_forall", None)

        reserved_keys = {
            "generate_variant_forall",
        }

        for variant in params_dict["shader_variants"]:
            default_iterated_params_names = set(
                default_iterated_params.keys()
                if default_iterated_params is not None
                else {}
            )
            variant_params_names = set(variant.keys())

            invalid_keys = (
                variant_params_names
                - default_iterated_params_names
                - params_names
                - reserved_keys
            )
            if invalid_keys:
                raise ValueError(f"unknown variant key(s): {sorted(invalid_keys)}")

            iterated_params = variant.get(
                "generate_variant_forall", default_iterated_params
            )

            if iterated_params is not None:
                variant_combinations = generate_variant_combinations(
                    iterated_params, variant_params_names
                )

                for combination in variant_combinations:
                    default_params_copy = copy.deepcopy(default_params)
                    for key in variant:
                        if key not in reserved_keys:
                            default_params_copy[key] = variant[key]

                    variant_name = variant["NAME"]

                    for setting in combination:
                        param_names = setting[0]
                        suffix = setting[1]
                        param_values = setting[2]
                        if isinstance(param_names, list):
                            for param_name, param_value in zip(
                                param_names, param_values
                            ):
                                default_params_copy[param_name] = param_value
                        else:
                            default_params_copy[param_names] = param_values

                        if len(str(suffix)) > 0:
                            variant_name = f"{variant_name}_{suffix}"

                    default_params_copy["NAME"] = variant_name
                    default_params_copy["VARIANT_NAME"] = variant["NAME"]

                    shader_template_params[template_name].append(default_params_copy)
            else:
                default_params_copy = copy.deepcopy(default_params)
                for key in variant:
                    if key not in reserved_keys:
                        default_params_copy[key] = variant[key]

                shader_template_params[template_name].append(default_params_copy)

    return shader_template_params


def symbol_base(stem: str) -> str:
    """snake_case shader stem -> PascalCase symbol base (binary_add -> BinaryAdd)."""
    return "".join(part.capitalize() for part in stem.split("_"))


_INT_LITERAL_RE = re.compile(r"^(\d+)[uUiI]?$")


def _resolve_dim(tok: str, src: str) -> int:
    """Resolve one @workgroup_size dim token: a literal or an override/const ident.

    Accepts WGSL suffix-typed integer literals (e.g. `64u`, `64i`) both as the
    token and on the right-hand side of an `override`/`const` (type optional).
    """
    lit = _INT_LITERAL_RE.match(tok)
    if lit:
        return int(lit.group(1))
    m = re.search(
        r"(?:override|const)\s+"
        + re.escape(tok)
        + r"\s*(?::\s*u32\s*)?=\s*(\d+)[uUiI]?",
        src,
    )
    if not m:
        raise ValueError(f"cannot resolve @workgroup_size identifier '{tok}'")
    return int(m.group(1))


def parse_workgroup_size(src: str) -> tuple[int, int, int]:
    """Resolve the (x, y, z) dims of @workgroup_size; y and z default to 1."""
    m = re.search(r"@workgroup_size\s*\(([^)]*)\)", src)
    if not m:
        raise ValueError("no @workgroup_size found")
    toks = [t.strip() for t in m.group(1).split(",") if t.strip()]
    if not toks or len(toks) > 3:
        raise ValueError(f"@workgroup_size takes 1-3 dims, got {len(toks)}")
    dims = [_resolve_dim(t, src) for t in toks]
    while len(dims) < 3:
        dims.append(1)
    return (dims[0], dims[1], dims[2])


def wgsl_sha256(wgsl_text: str) -> str:
    return hashlib.sha256(wgsl_text.encode("utf-8")).hexdigest()


def embedded_sha256(header_text: str) -> str:
    m = _SHA_RE.search(header_text)
    return m.group(1) if m else ""


def _wg_size_const(base: str, axis: str, val: int) -> str:
    """One WorkgroupSize constant; wrap to <=80 cols so CLANGFORMAT accepts it.

    Long shader names push the single-line form past the 80-col limit. Emit the
    wrapped form that clang-format selects so generated headers stay byte-stable.
    """
    name = f"k{base}WorkgroupSize{axis}"
    prefix = f"inline constexpr uint32_t {name} ="
    decl = f"{prefix} {val};"
    if len(decl) > 85:
        return f"inline constexpr uint32_t\n    {name} = {val};\n"
    if len(decl) > 80:
        return f"{prefix}\n    {val};\n"
    return f"{decl}\n"


def render_header(
    name_or_path, wgsl_text: str, provenance_stem: Optional[str] = None
) -> str:
    """Render the full <name>_wgsl.h text for a shader (shader embedded unchanged).

    Two call forms:
      - render_header(wgsl_path, wgsl_text): the plain, non-templated shaders --
        the symbol base and the `// @generated from` filename both derive from
        Path(wgsl_path).stem.
      - render_header(name, wgsl_text, provenance_stem): `name` is an expanded
        variant name that drives the emitted symbols; `provenance_stem` is the
        template stem cited in the `// @generated from` line.
    """
    if provenance_stem is None:
        name = Path(name_or_path).stem
        provenance_stem = name
    else:
        name = name_or_path
    if ')"' in wgsl_text:
        raise ValueError('shader contains )" which would close the R"( literal')
    base = symbol_base(name)
    x, y, z = parse_workgroup_size(wgsl_text)
    provenance = f"// @generated from {provenance_stem}.wgsl - DO NOT EDIT."
    if len(provenance) > 80:
        provenance_lines = [
            f"// @generated from {provenance_stem}.wgsl",
            "// DO NOT EDIT.",
        ]
    else:
        provenance_lines = [provenance]

    head = [
        _BSD_HEADER,
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace executorch::backends::webgpu {",
        "",
        *provenance_lines,
        f"// wgsl-sha256: {wgsl_sha256(wgsl_text)}",
        f'inline constexpr const char* k{base}WGSL = R"(',
    ]
    return (
        "\n".join(head)
        + "\n"
        + wgsl_text
        + ')";'
        + "\n\n"
        + _wg_size_const(base, "X", x)
        + _wg_size_const(base, "Y", y)
        + _wg_size_const(base, "Z", z)
        + "\n"
        + "} // namespace executorch::backends::webgpu\n"
    )


def discover():
    """All shader sources under runtime/ops, sorted."""
    return sorted((BACKEND_ROOT / "runtime/ops").glob("**/*.wgsl"))


class RegistryEntry(NamedTuple):
    name: str
    include: str
    symbol: str


def registry_path() -> Path:
    return BACKEND_ROOT / "runtime/WebGPUShaderRegistry.cpp"


def _registry_entry(header: Path) -> RegistryEntry:
    suffix = "_wgsl.h"
    if not header.name.endswith(suffix):
        raise ValueError(f"unexpected generated header name: {header.name}")
    name = header.name[: -len(suffix)]
    return RegistryEntry(
        name=name,
        include=header.relative_to(BACKEND_ROOT).as_posix(),
        symbol=symbol_base(name),
    )


def _collect_header_outputs() -> Tuple[Dict[Path, str], List[RegistryEntry]]:
    """Render every concrete header once and reject global collisions."""
    outputs: Dict[Path, str] = {}
    entries: List[RegistryEntry] = []
    registry_names: Set[str] = set()
    registry_symbols: Set[str] = set()
    for wgsl in discover():
        try:
            rendered_headers = list(headers_for_shader(wgsl))
        except Exception as error:
            raise ValueError(f"{wgsl.relative_to(BACKEND_ROOT)}: {error}") from error
        for header, rendered in rendered_headers:
            if header in outputs:
                raise ValueError(
                    "duplicate generated header path: "
                    f"{header.relative_to(BACKEND_ROOT)}"
                )
            entry = _registry_entry(header)
            if entry.name in registry_names:
                raise ValueError(f"duplicate shader registry name: {entry.name}")
            if entry.symbol in registry_symbols:
                raise ValueError(f"duplicate shader registry symbol: {entry.symbol}")
            outputs[header] = rendered
            entries.append(entry)
            registry_names.add(entry.name)
            registry_symbols.add(entry.symbol)
    return outputs, sorted(entries)


def registry_entries() -> List[RegistryEntry]:
    """Return one registry entry for every concrete generated shader."""
    _, entries = _collect_header_outputs()
    return entries


def render_registry(entries: List[RegistryEntry]) -> str:
    """Render the generated name-to-WGSL registry implementation."""
    ordered = sorted(entries)
    names = [entry.name for entry in ordered]
    if len(names) != len(set(names)):
        raise ValueError("duplicate shader registry name")

    includes = "\n".join(
        sorted(
            f"#include <executorch/backends/webgpu/{entry.include}>"
            for entry in ordered
        )
    )
    values = "\n".join(
        "    {\n"
        f'        "{entry.name}",\n'
        f"        k{entry.symbol}WGSL,\n"
        f"        k{entry.symbol}WorkgroupSizeX,\n"
        f"        k{entry.symbol}WorkgroupSizeY,\n"
        f"        k{entry.symbol}WorkgroupSizeZ,\n"
        "    },"
        for entry in ordered
    )
    return f"""{_BSD_HEADER}

// @generated by scripts/gen_wgsl_headers.py - DO NOT EDIT.

#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>

{includes}

#include <array>
#include <stdexcept>
#include <string>

namespace executorch::backends::webgpu {{
namespace {{

constexpr std::array<WebGPUShaderInfo, {len(ordered)}> kShaderRegistry = {{{{
{values}
}}}};

}} // namespace

const WebGPUShaderInfo& get_webgpu_shader_info(std::string_view name) {{
  for (const auto& shader : kShaderRegistry) {{
    if (shader.name == name) {{
      return shader;
    }}
  }}
  throw std::runtime_error(
      "WebGPU shader registry: unknown shader '" + std::string(name) + "'");
}}

}} // namespace executorch::backends::webgpu
"""


def headers_for_shader(wgsl):
    """Yield (header_path, rendered_text) pairs for one shader source.

    A shader is a template iff a sibling <stem>.yaml spec exists: each expanded
    variant emits its own <NAME>_wgsl.h (the provenance line cites the template
    stem). Otherwise the shader is embedded unchanged into <stem>_wgsl.h.
    """
    stem = wgsl.stem
    text = wgsl.read_text()
    spec_path = wgsl.with_name(stem + ".yaml")
    if spec_path.exists():
        spec = parse_template_spec(spec_path)
        if list(spec.keys()) != [stem]:
            raise ValueError(
                f"{spec_path.name}: top-level key must be '{stem}', got {list(spec.keys())}"
            )
        for variant_params in spec[stem]:
            name = variant_params["NAME"]
            expanded = preprocess(text, {**WGSL_HELPERS, **variant_params})
            header = wgsl.with_name(name + "_wgsl.h")
            yield header, render_header(name, expanded, stem)
    else:
        if "$if " in text or "${" in text:
            raise ValueError(
                f"shader uses $if/${{ templating but has no sibling {stem}.yaml spec"
            )
        header = wgsl.with_name(stem + "_wgsl.h")
        yield header, render_header(stem, text, stem)


def collect_outputs() -> Tuple[Dict[Path, bytes], List[Path]]:
    """Render the complete output tree and report unexpected old headers."""
    header_outputs, entries = _collect_header_outputs()
    outputs = {
        path: rendered.encode("utf-8") for path, rendered in header_outputs.items()
    }
    registry = registry_path()
    if registry in outputs:
        raise ValueError(f"duplicate generated output path: {registry}")
    outputs[registry] = render_registry(entries).encode("utf-8")

    expected_headers = set(header_outputs)
    actual_headers = set((BACKEND_ROOT / "runtime/ops").glob("**/*_wgsl.h"))
    return outputs, sorted(actual_headers - expected_headers)


class _OriginalOutput(NamedTuple):
    existed: bool
    contents: bytes
    mode: int


def _stage_bytes(destination: Path, contents: bytes, mode: int) -> Path:
    """Write one same-directory candidate without changing its destination."""
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.wgsl-gen-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(contents)
        temporary.chmod(mode)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return temporary


def _cleanup_temporaries(temporaries) -> List[str]:
    errors = []
    for temporary in temporaries:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"cannot remove temporary {temporary}: {error}")
    return errors


def _stage_outputs(
    outputs: Dict[Path, bytes], changed: List[Path]
) -> Tuple[Dict[Path, _OriginalOutput], Dict[Path, Path], List[str]]:
    originals: Dict[Path, _OriginalOutput] = {}
    staged: Dict[Path, Path] = {}
    try:
        for destination in sorted(changed):
            if destination.exists():
                original = _OriginalOutput(
                    existed=True,
                    contents=destination.read_bytes(),
                    mode=stat.S_IMODE(destination.stat().st_mode),
                )
            else:
                original = _OriginalOutput(False, b"", 0o644)
            originals[destination] = original
            staged[destination] = _stage_bytes(
                destination, outputs[destination], original.mode
            )
    except BaseException as error:
        cleanup_errors = _cleanup_temporaries(staged.values())
        if isinstance(error, OSError):
            errors = [f"cannot stage generated output: {error}"] + cleanup_errors
            return originals, staged, errors
        raise
    return originals, staged, []


def _rollback_outputs(
    originals: Dict[Path, _OriginalOutput],
    replaced: List[Path],
    staged: Dict[Path, Path],
) -> List[str]:
    errors = []
    for destination in reversed(replaced):
        original = originals[destination]
        restore_temporary: Optional[Path] = None
        try:
            if original.existed:
                restore_temporary = _stage_bytes(
                    destination, original.contents, original.mode
                )
                os.replace(restore_temporary, destination)
            else:
                destination.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"cannot roll back {destination}: {error}")
        finally:
            if restore_temporary is not None:
                errors.extend(_cleanup_temporaries([restore_temporary]))
    errors.extend(_cleanup_temporaries(staged.values()))
    return errors


def _publish_outputs(outputs: Dict[Path, bytes], changed: List[Path]) -> List[str]:
    """Stage and publish changed outputs, rolling back reported failures."""
    originals, staged, stage_errors = _stage_outputs(outputs, changed)
    if stage_errors:
        return stage_errors

    replaced: List[Path] = []
    try:
        for destination in sorted(changed):
            try:
                os.replace(staged[destination], destination)
            except OSError:
                raise
            except BaseException:
                replaced.append(destination)
                raise
            else:
                replaced.append(destination)
    except OSError as commit_error:
        return [f"cannot publish generated output: {commit_error}"] + _rollback_outputs(
            originals, replaced, staged
        )
    except BaseException:
        _rollback_outputs(originals, replaced, staged)
        raise

    return _cleanup_temporaries(staged.values())


def _report_drift(missing, stale, orphans) -> None:
    """Print the --check report for missing/stale committed headers."""
    if missing:
        print("Missing embedded WGSL headers (run scripts/gen_wgsl_headers.py):")
        for h in missing:
            print(f"  {h.relative_to(BACKEND_ROOT)}")
    if stale:
        print("Stale embedded WGSL headers (run scripts/gen_wgsl_headers.py):")
        for h in stale:
            print(f"  {h.relative_to(BACKEND_ROOT)}")
    if orphans:
        print("Orphan embedded WGSL headers (remove or restore their sources):")
        for h in orphans:
            print(f"  {h.relative_to(BACKEND_ROOT)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed headers match (exit 1 on drift)",
    )
    args = parser.parse_args(argv)

    try:
        outputs, orphans = collect_outputs()
        missing = []
        stale = []
        for output, want in sorted(outputs.items()):
            if not output.exists():
                missing.append(output)
            elif output.read_bytes() != want:
                stale.append(output)
    except Exception as error:
        print("Cannot generate WGSL outputs:")
        print(f"  {error}")
        return 1

    if orphans:
        _report_drift([], [], orphans)
        return 1

    if args.check:
        if stale or missing:
            _report_drift(missing, stale, [])
            return 1
        return 0

    errors = _publish_outputs(outputs, missing + stale)
    if errors:
        print("Cannot publish WGSL outputs:")
        for error in errors:
            print(f"  {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
