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

A shader is treated as a template iff a sibling <stem>.json spec exists; the
$-block engine (preprocess/escape/generate_variant_combinations) expands one
template + a DTYPE/VEC variant matrix into the concrete per-variant headers.

Stdlib only (the devserver has no third-party pip; no yaml).
"""

import argparse
import copy
import hashlib
import io
import json
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
#  parse_template_spec) expand one template + its JSON sidecar into the
#  per-variant WGSL headers.
########################################################################


# WGSL type-helpers injected into preprocess's exec globals so ${...} template
# expressions can spell WGSL types (f32/f16, vec4<T>); @group/@binding layout is
# written directly in the templates.
def wgsl_scalar_type(dtype: str) -> str:
    if dtype == "half":
        return "f16"
    elif dtype == "float":
        return "f32"
    return dtype


def wgsl_buffer_type(dtype: str, vec: int) -> str:
    if vec == 1:
        return wgsl_scalar_type(dtype)
    return f"vec{vec}<{wgsl_scalar_type(dtype)}>"


def wgsl_accum_type() -> str:
    # Accumulators stay f32 in every variant (f16 accumulation is numerically
    # unsafe on target GPUs).
    return "f32"


WGSL_HELPERS: Dict[str, Any] = {
    "wgsl_scalar_type": wgsl_scalar_type,
    "wgsl_buffer_type": wgsl_buffer_type,
    "wgsl_accum_type": wgsl_accum_type,
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


# json object_pairs_hook that rejects duplicate keys in a spec object.
def _reject_duplicate_keys(pairs: List[Any]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"found duplicate key: {key!r}")
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


def parse_template_spec(json_path) -> Dict[str, List[Dict[str, Any]]]:  # noqa: C901
    """Parse a <stem>.json variant spec into {template_name: [expanded
    per-variant param dicts]}. Stdlib JSON with a dup-key-rejecting
    object_pairs_hook."""
    shader_template_params: Dict[str, List[Dict[str, Any]]] = {}
    with open(json_path) as f:
        contents = json.load(f, object_pairs_hook=_reject_duplicate_keys)
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
            assert len(invalid_keys) == 0

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


def render_header(name_or_path, wgsl_text: str, provenance_stem: str = None) -> str:
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

    head = [
        _BSD_HEADER,
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace executorch::backends::webgpu {",
        "",
        f"// @generated from {provenance_stem}.wgsl - DO NOT EDIT.",
        f"// wgsl-sha256: {wgsl_sha256(wgsl_text)}",
        f'inline constexpr const char* k{base}WGSL = R"(',
    ]
    return (
        "\n".join(head)
        + "\n"
        + wgsl_text
        + ')";'
        + "\n\n"
        + f"inline constexpr uint32_t k{base}WorkgroupSizeX = {x};\n"
        + f"inline constexpr uint32_t k{base}WorkgroupSizeY = {y};\n"
        + f"inline constexpr uint32_t k{base}WorkgroupSizeZ = {z};\n\n"
        + "} // namespace executorch::backends::webgpu\n"
    )


def discover():
    """All shader sources under runtime/ops, sorted."""
    return sorted((BACKEND_ROOT / "runtime/ops").glob("**/*.wgsl"))


def headers_for_shader(wgsl):
    """Yield (header_path, rendered_text) pairs for one shader source.

    A shader is a template iff a sibling <stem>.json spec exists: each expanded
    variant emits its own <NAME>_wgsl.h (the provenance line cites the template
    stem). Otherwise the shader is embedded unchanged into <stem>_wgsl.h.
    """
    stem = wgsl.stem
    text = wgsl.read_text()
    spec_path = wgsl.with_name(stem + ".json")
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
                f"shader uses $if/${{ templating but has no sibling {stem}.json spec"
            )
        header = wgsl.with_name(stem + "_wgsl.h")
        yield header, render_header(stem, text, stem)


def _report_drift(missing, stale) -> None:
    """Print the --check report for missing/stale committed headers."""
    if missing:
        print("Missing embedded WGSL headers (run scripts/gen_wgsl_headers.py):")
        for h in missing:
            print(f"  {h.relative_to(BACKEND_ROOT)}")
    if stale:
        print("Stale embedded WGSL headers (run scripts/gen_wgsl_headers.py):")
        for h in stale:
            print(f"  {h.relative_to(BACKEND_ROOT)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed headers match (exit 1 on drift)",
    )
    args = parser.parse_args(argv)

    stale = []
    missing = []
    errors = []
    for wgsl in discover():
        try:
            rendered = list(headers_for_shader(wgsl))
        except ValueError as e:
            errors.append(f"{wgsl.relative_to(BACKEND_ROOT)}: {e}")
            continue
        for header, want in rendered:
            # Full-content compare (not just the sha) catches generator-logic drift too.
            if header.exists() and header.read_text() == want:
                continue
            if args.check:
                (missing if not header.exists() else stale).append(header)
            else:
                header.write_text(want)

    if errors:
        print("Cannot generate header (malformed shader):")
        for e in errors:
            print(f"  {e}")
        return 1
    if args.check and (stale or missing):
        _report_drift(missing, stale)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
