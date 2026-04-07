#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from executorch.devtools.bundled_program.serialize import (
    deserialize_from_flatbuffer_to_bundled_program,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.backend.compile_spec_schema import CompileSpec


@dataclass(frozen=True)
class DelegateInfo:
    plan_index: int
    plan_name: str
    delegate_index: int
    delegate_id: str
    compile_specs: List[CompileSpec]


def _extract_pte_from_bundle(pte_data: bytes) -> bytes:
    try:
        bundled = deserialize_from_flatbuffer_to_bundled_program(pte_data)
    except Exception:
        return pte_data
    return bundled.program if bundled.program else pte_data


def _decode_compile_spec_value(value: bytes) -> str | None:
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if all(char.isprintable() or char in "\r\n\t" for char in decoded):
        return decoded
    return None


def _compile_spec_to_dict(spec: CompileSpec) -> dict[str, Any]:
    value_text = _decode_compile_spec_value(spec.value)
    return {
        "key": spec.key,
        "value_text": value_text,
        "value_hex": spec.value.hex(),
    }


def delegate_info_to_dict(delegate_info: DelegateInfo) -> dict[str, Any]:
    return {
        "plan_index": delegate_info.plan_index,
        "plan_name": delegate_info.plan_name,
        "delegate_index": delegate_info.delegate_index,
        "delegate_id": delegate_info.delegate_id,
        "compile_specs": [
            _compile_spec_to_dict(spec) for spec in delegate_info.compile_specs
        ],
    }


def get_delegate_infos_from_pte(pte_path: str | Path) -> list[DelegateInfo]:
    pte_path = Path(pte_path)
    pte_data = pte_path.read_bytes()
    if pte_path.suffix.lower() == ".bpte":
        pte_data = _extract_pte_from_bundle(pte_data)
    program = deserialize_pte_binary(pte_data).program

    delegate_infos: list[DelegateInfo] = []
    for plan_index, plan in enumerate(program.execution_plan):
        for delegate_index, delegate in enumerate(plan.delegates):
            delegate_infos.append(
                DelegateInfo(
                    plan_index=plan_index,
                    plan_name=plan.name,
                    delegate_index=delegate_index,
                    delegate_id=delegate.id,
                    compile_specs=list(delegate.compile_specs),
                )
            )
    return delegate_infos


def format_delegate_infos(
    delegate_infos: list[DelegateInfo], output_format: str = "pretty"
) -> str:
    if output_format == "json":
        return json.dumps(
            [delegate_info_to_dict(delegate_info) for delegate_info in delegate_infos],
            indent=2,
            sort_keys=True,
        )

    lines = []
    for delegate_info in delegate_infos:
        lines.append(
            f"plan {delegate_info.plan_index} {delegate_info.plan_name}, "
            f"delegate {delegate_info.delegate_index} {delegate_info.delegate_id}:"
        )
        for spec in delegate_info.compile_specs:
            spec_dict = _compile_spec_to_dict(spec)
            rendered_value = (
                json.dumps(spec_dict["value_text"])
                if spec_dict["value_text"] is not None
                else f"0x{spec_dict['value_hex']}"
            )
            lines.append(f"  {spec_dict['key']}={rendered_value}")
    return "\n".join(lines)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print all delegate compile_specs from a .pte or .bpte file."
    )
    parser.add_argument("pte_path", help="Path to a .pte or .bpte file.")
    parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format. Default: %(default)s.",
    )
    return parser.parse_args()


def main() -> int:
    args = _get_args()
    delegate_infos = get_delegate_infos_from_pte(args.pte_path)
    if not delegate_infos:
        return 2

    print(format_delegate_infos(delegate_infos, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
