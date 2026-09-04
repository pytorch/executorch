# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""P0 boundary diagnostics for the VGF partitioner.

The runtime delegate cannot see portable operations immediately outside a VGF
partition. This module records those operations after partition tagging so q/dq,
layout conversions and materializing copies are not silently attributed to the
VGF runtime or incorrectly reported as zero.

"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.fx import GraphModule, Node

logger = logging.getLogger(__name__)

_DIAGNOSTICS_DIR_ENV = "EXECUTORCH_VGF_DIAGNOSTICS_DIR"
_SOURCE_REVISION_ENV = "EXECUTORCH_VGF_SOURCE_REVISION"


def _target_name(node: Node) -> str:
    target = node.target
    # OpOverload stringification is stable. For ordinary Python callables,
    # avoid repr() because it embeds a process-specific memory address.
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(target)


def _classify_conversion(node: Node) -> str | None:
    if node.op != "call_function":
        return None

    target = _target_name(node).lower()
    if "dequantize" in target:
        return "DEQUANTIZE"
    if "quantize" in target:
        return "QUANTIZE"

    layout_markers = (
        "_to_dim_order_copy",
        "_clone_dim_order",
        "permute_copy",
        "transpose_copy",
        "contiguous",
        "memory_format",
    )
    if any(marker in target for marker in layout_markers):
        return "LAYOUT_CONVERSION"

    # Keep this list explicit. These operators can materialize storage at a
    # delegate boundary even when their logical value is view-like.
    copy_markers = (
        "clone",
        "_to_copy",
        "alias_copy",
        "detach_copy",
        "as_strided_copy",
        "view_copy",
        "reshape_copy",
        "expand_copy",
        "slice_copy",
        "select_copy",
        "copy.default",
    )
    if any(marker in target for marker in copy_markers):
        return "MATERIALIZING_COPY"
    return None


def _static_int(value: Any) -> int | str:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        # SymInt can be concretized only when statically known.
        return int(value)
    except (TypeError, ValueError, RuntimeError):
        return str(value)


def _tensor_metadata(node: Node) -> dict[str, Any]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return {
            "dtype": None,
            "shape": None,
            "strides": None,
            "logical_bytes": None,
        }

    shape = [_static_int(dim) for dim in value.shape]
    strides = [_static_int(stride) for stride in value.stride()]
    logical_bytes: int | None = None
    if all(isinstance(dim, int) and dim >= 0 for dim in shape):
        numel = 1
        for dim in shape:
            assert isinstance(dim, int)
            numel *= dim
        logical_bytes = numel * value.element_size()

    return {
        "dtype": str(value.dtype),
        "shape": shape,
        "strides": strides,
        "logical_bytes": logical_bytes,
    }


def _literal_parameters(node: Node) -> list[Any]:
    """Return JSON-safe non-Node arguments useful for q/dq diagnostics."""

    def make_json_safe(value: Any) -> Any:
        if isinstance(value, Node):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, (tuple, list)):
            return [make_json_safe(item) for item in value]
        return str(value)

    params: list[Any] = []
    for arg in node.args:
        if isinstance(arg, Node):
            continue
        params.append(make_json_safe(arg))
    if node.kwargs:
        params.append(
            {key: make_json_safe(value) for key, value in node.kwargs.items()}
        )
    return params


def _conversion_record(module_path: str, node: Node) -> dict[str, Any] | None:
    kind = _classify_conversion(node)
    if kind is None:
        return None

    input_meta = None
    for input_node in node.all_input_nodes:
        input_meta = _tensor_metadata(input_node)
        break

    return {
        "module_path": module_path,
        "node": node.name,
        "op": _target_name(node),
        "kind": kind,
        "input_tensor": input_meta,
        "output_tensor": _tensor_metadata(node),
        "literal_parameters": _literal_parameters(node),
    }


def _walk_input_conversion_chain(module_path: str, start: Node) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current: Node | None = start
    seen: set[Node] = set()
    while current is not None and current not in seen:
        seen.add(current)
        record = _conversion_record(module_path, current)
        if record is None:
            break
        chain.append(record)
        current = current.all_input_nodes[0] if current.all_input_nodes else None
    return chain


def _walk_output_conversion_chain(
    module_path: str, start: Node
) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current: Node | None = start
    seen: set[Node] = set()
    while current is not None and current not in seen:
        seen.add(current)
        record = _conversion_record(module_path, current)
        if record is None:
            break
        chain.append(record)
        # Continue only through an unambiguous linear boundary conversion chain.
        users = list(current.users)
        current = users[0] if len(users) == 1 else None
    return chain


def _tagged(node: Node, tag: str) -> bool:
    return node.meta.get("delegation_tag") == tag


def _boundary_input_record(
    module_path: str,
    tag: str,
    external: Node,
    internal: Node,
) -> dict[str, Any]:
    return {
        "external_node": external.name,
        "external_op": (
            external.op if external.op != "call_function" else _target_name(external)
        ),
        "delegate_node": internal.name,
        "tensor": _tensor_metadata(external),
        "method_input": external.op == "placeholder",
        "fan_out": any(not _tagged(user, tag) for user in external.users),
        "conversion_chain": _walk_input_conversion_chain(module_path, external),
    }


def _boundary_output_record(
    module_path: str,
    tag: str,
    internal: Node,
    external: Node,
) -> dict[str, Any]:
    return {
        "delegate_node": internal.name,
        "external_node": external.name,
        "external_op": (
            external.op if external.op != "call_function" else _target_name(external)
        ),
        "tensor": _tensor_metadata(internal),
        "method_output": external.op == "output",
        "fan_out": len(internal.users) > 1,
        "conversion_chain": _walk_output_conversion_chain(module_path, external),
    }


def collect_vgf_boundary_manifest(  # noqa: C901
    graph_module: GraphModule,
    partition_tags: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect method/delegate boundary evidence after partition tagging."""
    partitions: list[dict[str, Any]] = []

    for module_path, module in graph_module.named_modules():
        if not isinstance(module, GraphModule):
            continue
        nodes = list(module.graph.nodes)
        for tag in sorted(partition_tags):
            tagged_nodes = [node for node in nodes if _tagged(node, tag)]
            if not tagged_nodes:
                continue

            input_records: list[dict[str, Any]] = []
            output_records: list[dict[str, Any]] = []
            seen_inputs: set[tuple[str, str]] = set()
            seen_outputs: set[tuple[str, str]] = set()

            for node in tagged_nodes:
                for input_node in node.all_input_nodes:
                    if _tagged(input_node, tag):
                        continue
                    key = (input_node.name, node.name)
                    if key in seen_inputs:
                        continue
                    seen_inputs.add(key)
                    input_records.append(
                        _boundary_input_record(module_path, tag, input_node, node)
                    )

                for user in node.users:
                    if _tagged(user, tag):
                        continue
                    key = (node.name, user.name)
                    if key in seen_outputs:
                        continue
                    seen_outputs.add(key)
                    output_records.append(
                        _boundary_output_record(module_path, tag, node, user)
                    )

            partitions.append(
                {
                    "tag": tag,
                    "module_path": module_path or "<root>",
                    "node_count": len(tagged_nodes),
                    "inputs": input_records,
                    "outputs": output_records,
                }
            )

    # Aggregate unique conversion nodes so fan-out does not double-count bytes.
    unique_conversions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for partition in partitions:
        for boundary in (*partition["inputs"], *partition["outputs"]):
            for conversion in boundary["conversion_chain"]:
                conversion_key: tuple[str, str, str] = (
                    conversion["module_path"],
                    conversion["node"],
                    conversion["kind"],
                )
                unique_conversions[conversion_key] = conversion

    by_kind: dict[str, dict[str, int]] = {}
    for conversion in unique_conversions.values():
        stats = by_kind.setdefault(
            conversion["kind"],
            {
                "count": 0,
                "known_input_bytes": 0,
                "known_output_bytes": 0,
                "unknown_input_byte_count": 0,
                "unknown_output_byte_count": 0,
            },
        )
        stats["count"] += 1
        input_tensor = conversion["input_tensor"]
        input_bytes = None if input_tensor is None else input_tensor["logical_bytes"]
        output_bytes = conversion["output_tensor"]["logical_bytes"]
        if input_bytes is None:
            stats["unknown_input_byte_count"] += 1
        else:
            stats["known_input_bytes"] += input_bytes
        if output_bytes is None:
            stats["unknown_output_byte_count"] += 1
        else:
            stats["known_output_bytes"] += output_bytes

    unclassified_boundary_ops: list[dict[str, Any]] = []
    for partition in partitions:
        for direction, boundaries in (
            ("INPUT", partition["inputs"]),
            ("OUTPUT", partition["outputs"]),
        ):
            for boundary in boundaries:
                external_op = boundary["external_op"]
                if external_op in {"placeholder", "output"}:
                    continue
                if boundary["conversion_chain"]:
                    continue
                unclassified_boundary_ops.append(
                    {
                        "partition_tag": partition["tag"],
                        "module_path": partition["module_path"],
                        "direction": direction,
                        "external_node": boundary["external_node"],
                        "external_op": external_op,
                        "tensor": boundary["tensor"],
                        "reason": "external boundary op is not a classified q/dq/layout/materializing-copy transform",
                    }
                )

    return {
        "schema": "executorch.vgf.aot_boundary_manifest",
        "schema_version": 1,
        "source_revision": os.environ.get(_SOURCE_REVISION_ENV),
        "analysis_phase": "post_partition_tagging",
        "partitions": partitions,
        "summary": {
            "partition_count": len(partitions),
            "unique_boundary_conversion_count": len(unique_conversions),
            "conversions_by_kind": by_kind,
            "unclassified_external_boundary_op_count": len(unclassified_boundary_ops),
            "unclassified_external_boundary_ops": unclassified_boundary_ops,
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_vgf_boundary_manifest_if_requested(
    graph_module: GraphModule,
    partition_tags: Mapping[str, Any],
    intermediate_path: str | None,
) -> list[str]:
    """Write the boundary manifest to configured diagnostic destinations."""
    destinations: list[Path] = []
    env_path = os.environ.get(_DIAGNOSTICS_DIR_ENV)
    if env_path:
        destinations.append(Path(env_path))
    if intermediate_path:
        destinations.append(Path(intermediate_path))

    # Preserve order while avoiding duplicate writes to the same directory.
    unique_destinations = list(dict.fromkeys(destinations))
    if not unique_destinations:
        return []

    manifest = collect_vgf_boundary_manifest(graph_module, partition_tags)
    manifest_id = uuid.uuid4().hex
    manifest["manifest_id"] = manifest_id

    written: list[str] = []
    for destination in unique_destinations:
        path = destination / f"aot_boundary_manifest.{manifest_id}.json"
        try:
            _atomic_write_json(path, manifest)
        except OSError as exc:
            logger.warning("Failed to write VGF boundary manifest %s: %s", path, exc)
            continue
        written.append(str(path))
        logger.info("Wrote VGF boundary manifest to %s", path)
    return written
