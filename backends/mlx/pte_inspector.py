#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PTE Inspector - Extract and dump data from ExecuTorch .pte files.

This utility can:
1. Parse the PTE file structure (header, flatbuffer, segments)
2. Extract delegate payloads (e.g., MLX backend data)
3. Convert FlatBuffer data to JSON for inspection

Usage:
    python pte_inspector.py mlx_mlp.pte
    python pte_inspector.py mlx_mlp.pte --output output.json
    python pte_inspector.py mlx_mlp.pte --extract-delegate mlx --output mlx_payload.bin
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from executorch.backends.mlx._generated_inspector import OP_NODE_FIELDS
from executorch.backends.mlx.serialization._generated_serializers import (
    MLX_OP_TYPE_NAMES,
)
from executorch.exir._serialize._program import (
    _ExtendedHeader,
    _extract_delegate_payload as extract_delegate_payload,
)

MLX_MAGIC = b"MLX0"
MLX_HEADER_LENGTH = 24

_SLOT_TYPE_NAMES = {0: "Tensor", 1: "Int", 2: "Float", 3: "Bool"}


@dataclass
class MLXHeader:

    magic: bytes
    data_segment_offset: int
    data_segment_size: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "MLXHeader":
        if len(data) < MLX_HEADER_LENGTH:
            raise ValueError(
                f"Not enough data for MLX header: {len(data)} < {MLX_HEADER_LENGTH}"
            )

        # Layout: [4 bytes padding][4 bytes magic][8 bytes offset][8 bytes size]
        magic = data[4:8]
        data_segment_offset = int.from_bytes(data[8:16], byteorder="little")
        data_segment_size = int.from_bytes(data[16:24], byteorder="little")

        return cls(
            magic=magic,
            data_segment_offset=data_segment_offset,
            data_segment_size=data_segment_size,
        )

    def is_valid(self) -> bool:
        return self.magic == MLX_MAGIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magic": self.magic.decode("utf-8", errors="replace"),
            "data_segment_offset": self.data_segment_offset,
            "data_segment_size": self.data_segment_size,
        }


@dataclass
class MLXPayload:
    """Parsed MLX delegate payload: header + flatbuffer bytes."""

    header: MLXHeader
    fb_data: bytes
    raw: bytes


def _load_mlx_payload(pte_data: bytes, delegate_index: int = 0) -> MLXPayload:
    """Extract MLX delegate payload from PTE data and parse its header.

    Raises ``ValueError`` if the delegate cannot be found or the MLX header is
    invalid.
    """
    payload = extract_delegate_payload(pte_data, "mlx", delegate_index=delegate_index)
    if payload is None:
        raise ValueError(f"Could not extract MLX delegate {delegate_index}")

    header = MLXHeader.from_bytes(payload)
    if not header.is_valid():
        raise ValueError(f"Invalid MLX magic: {header.magic!r}")

    fb_data = payload[MLX_HEADER_LENGTH : header.data_segment_offset]
    return MLXPayload(header=header, fb_data=fb_data, raw=payload)


def _find_mlx_delegates(pte_data: bytes) -> List[Tuple[int, Dict]]:
    """Return list of ``(plan_index, delegate_dict)`` for every MLX delegate."""
    from executorch.exir._serialize._flatbuffer import _program_flatbuffer_to_json

    program_data = json.loads(_program_flatbuffer_to_json(pte_data))
    delegates: List[Tuple[int, Dict]] = []
    for plan in program_data.get("execution_plan", []):
        for i, delegate in enumerate(plan.get("delegates", [])):
            if "mlx" in delegate.get("id", "").lower():
                delegates.append((i, delegate))
    return delegates


def _get_fb_graph(fb_data: bytes):
    """Return the FlatBuffer MLXGraph root object."""
    from executorch.backends.mlx.serialization._generated.mlx_delegate import (
        MLXGraph as FBMLXGraph,
    )

    return FBMLXGraph.MLXGraph.GetRootAs(fb_data, 0)


def _parse_graph_info(graph) -> Dict[str, Any]:
    """Extract top-level graph scalars (tensor counts, chain counts, etc.)."""
    return {
        "version": graph.Version().decode("utf-8") if graph.Version() else None,
        "num_constant_tensors": graph.NumConstantTensors(),
        "num_input_tensors": graph.NumInputTensors(),
        "num_output_tensors": graph.NumOutputTensors(),
        "num_mutable_buffer_tensors": graph.NumMutableBufferTensors(),
        "num_temp_tensors": graph.NumTempTensors(),
        "num_values": graph.NumValues(),
        "num_instruction_chains": graph.InstructionChainsLength(),
        "main_chain_idx": graph.MainChainIdx(),
        "init_chain_idx": graph.InitChainIdx(),
        "input_map_length": graph.InputMapLength(),
        "output_map_length": graph.OutputMapLength(),
        "mutable_buffer_map_length": graph.MutableBufferMapLength(),
        "named_slots_length": graph.NamedSlotsLength(),
        "tensor_meta_length": graph.TensorMetaLength(),
    }


def _parse_instructions(graph) -> List[Dict[str, Any]]:
    """Parse all instruction chains and their op nodes."""
    chains: List[Dict[str, Any]] = []
    for c in range(graph.InstructionChainsLength()):
        chain = graph.InstructionChains(c)
        chain_info: Dict[str, Any] = {"chain_index": c, "instructions": []}
        if chain:
            for i in range(chain.InstructionsLength()):
                try:
                    instr = chain.Instructions(i)
                    if instr:
                        op_type = instr.OpType()
                        op_name = MLX_OP_TYPE_NAMES.get(op_type, f"Unknown({op_type})")
                        instr_info: Dict[str, Any] = {
                            "instr_idx": i,
                            "op_type": op_type,
                            "op_name": op_name,
                        }
                        op_data = _parse_op_node(instr, op_name)
                        if op_data:
                            instr_info.update(op_data)
                        chain_info["instructions"].append(instr_info)
                except Exception as e:
                    chain_info["instructions"].append(
                        {"instr_idx": i, "error": f"parse_failed: {e}"}
                    )
        chains.append(chain_info)
    return chains


def _parse_named_slots(graph) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    for i in range(graph.NamedSlotsLength()):
        try:
            ns = graph.NamedSlots(i)
            if ns:
                info: Dict[str, Any] = {
                    "name": ns.Name().decode("utf-8") if ns.Name() else None,
                }
                slot = ns.Slot()
                if slot:
                    info["slot_idx"] = slot.Idx()
                    info["slot_type"] = slot.SlotType()
                slots.append(info)
        except Exception as e:
            slots.append({"instr_idx": i, "error": f"parse_failed: {e}"})
    return slots


def _parse_tensor_meta(graph) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for i in range(graph.TensorMetaLength()):
        try:
            tm = graph.TensorMeta(i)
            if tm:
                shape: List[Any] = []
                for j in range(tm.ShapeLength()):
                    sd = tm.Shape(j)
                    if sd.Value() == -1:
                        lo = sd.MinValue()
                        hi = sd.MaxValue()
                        if hi == -1:
                            shape.append(f"dyn(min={lo})")
                        else:
                            shape.append(f"dyn({lo}..{hi})")
                    else:
                        shape.append(sd.Value())
                meta: Dict[str, Any] = {
                    "index": i,
                    "dtype": tm.Dtype(),
                    "shape": shape,
                }
                if tm.StridesLength() > 0:
                    meta["strides"] = [tm.Strides(j) for j in range(tm.StridesLength())]
                metas.append(meta)
        except Exception as e:
            metas.append({"instr_idx": i, "error": f"parse_failed: {e}"})
    return metas


def _parse_io_maps(
    graph,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Return (input_map, output_map, mutable_buffer_map) as slot-variant dicts."""

    def _extract(
        length_fn: Callable[[], int], getter_fn: Callable[[int], Any]
    ) -> List[Dict]:
        result = []
        for i in range(length_fn()):
            try:
                sv = getter_fn(i)
                if sv:
                    result.append({"idx": sv.Idx(), "slot_type": sv.SlotType()})
            except Exception as e:
                result.append({"instr_idx": i, "error": f"parse_failed: {e}"})
        return result

    return (
        _extract(graph.InputMapLength, graph.InputMap),
        _extract(graph.OutputMapLength, graph.OutputMap),
        _extract(graph.MutableBufferMapLength, graph.MutableBufferMap),
    )


def parse_mlx_flatbuffer(fb_data: bytes) -> Dict[str, Any]:
    """Parse MLX FlatBuffer data into a dict using the generated FlatBuffer bindings."""
    result: Dict[str, Any] = {}
    try:
        graph = _get_fb_graph(fb_data)

        result = _parse_graph_info(graph)
        result["instruction_chains"] = _parse_instructions(graph)
        result["named_slots"] = _parse_named_slots(graph)
        result["tensor_meta"] = _parse_tensor_meta(graph)

        input_map, output_map, mutable_buffer_map = _parse_io_maps(graph)
        result["input_map"] = input_map
        result["output_map"] = output_map
        result["mutable_buffer_map"] = mutable_buffer_map

        try:
            cs = graph.ConstantSegment()
            if cs:
                result["constant_segment"] = {
                    "offset": cs.Offset(),
                    "size": cs.Size(),
                }
        except Exception as e:
            result["constant_segment_error"] = f"parse_failed: {e}"

    except ImportError as e:
        result["error"] = f"FlatBuffer bindings not available: {e}"
        result["_fallback"] = "Using basic header parsing only"
    except Exception as e:
        result["error"] = f"FlatBuffer parse error: {e}"
        result["traceback"] = traceback.format_exc()

    return result


def _parse_op_node(instr, op_name: str) -> Optional[Dict[str, Any]]:
    """Parse the specific op node fields from an instruction.

    Uses the generated field mappings in ``OP_NODE_FIELDS`` to extract
    op-specific fields without manually maintaining per-op logic.
    """
    try:
        op = instr.Op()
        if op is None:
            return None

        if op_name not in OP_NODE_FIELDS:
            return {"error": f"Unknown op type: {op_name}"}

        module = __import__(
            f"executorch.backends.mlx.serialization._generated.mlx_delegate.{op_name}",
            fromlist=[op_name],
        )
        node_class = getattr(module, op_name)
        node = node_class()
        node.Init(op.Bytes, op.Pos)

        result: Dict[str, Any] = {}
        for field_name, accessor_name, kind in OP_NODE_FIELDS[op_name]:
            try:
                result[field_name] = _extract_field(node, accessor_name, kind)
            except Exception as e:
                result[field_name] = {"error": str(e)}

        result = {k: v for k, v in result.items() if v is not None}
        return result if result else None

    except Exception as e:
        return {"parse_error": str(e), "traceback": traceback.format_exc()}


def _extract_vid_or_tid(obj) -> Optional[Dict[str, Any]]:
    """Extract a VidOrTid FlatBuffer object into a dict.

    VidOrTid has: .IsVid() -> bool, .Vid() -> Vid|None, .Tid() -> Tid|None.
    Same pattern as IntOrVid but references value/tensor slots instead of
    holding a literal.
    """
    if obj is None:
        return None
    if obj.IsVid():
        v = obj.Vid()
        return {"vid": v.Idx()} if v else None
    t = obj.Tid()
    return {"tid": t.Idx()} if t else None


def _extract_field(node, accessor_name: str, kind: str) -> Any:  # noqa: C901
    """Extract a single field from a FlatBuffer op node based on its *kind*."""
    if kind == "tid":
        t = getattr(node, accessor_name)()
        return {"tid": t.Idx()} if t else None

    if kind == "vid":
        v = getattr(node, accessor_name)()
        return {"vid": v.Idx()} if v else None

    if kind == "vid_or_tid":
        return _extract_vid_or_tid(getattr(node, accessor_name)())

    if kind == "int_or_vid_or_tid":
        ivt = getattr(node, accessor_name)()
        if ivt is None:
            return None
        k = ivt.Kind()
        if k == 0:  # literal int
            return {"literal": ivt.Literal()}
        elif k == 1:  # Vid
            v = ivt.Vid()
            return {"vid": v.Idx()} if v else None
        elif k == 2:  # Tid
            t = ivt.Tid()
            return {"tid": t.Idx()} if t else None
        return {"kind": k}

    if kind == "int_or_vid":
        iov = getattr(node, accessor_name)()
        if iov is None:
            return None
        if iov.IsVid():
            v = iov.Vid()
            return {"vid": v.Idx()} if v else None
        return {"literal": iov.Literal()}

    if kind == "float_or_vid":
        fov = getattr(node, accessor_name)()
        if fov is None:
            return None
        if fov.IsVid():
            v = fov.Vid()
            return {"vid": v.Idx()} if v else None
        return {"literal": fov.Literal()}

    if kind == "int_list":
        length = getattr(node, f"{accessor_name}Length")()
        getter = getattr(node, accessor_name)
        return [getter(i) for i in range(length)]

    if kind == "tid_list":
        length = getattr(node, f"{accessor_name}Length")()
        getter = getattr(node, accessor_name)
        items = []
        for i in range(length):
            s = getter(i)
            items.append(f"tid {s.Idx()}" if s else None)
        return items

    if kind == "string_list":
        length = getattr(node, f"{accessor_name}Length")()
        getter = getattr(node, accessor_name)
        return [getter(i).decode("utf-8") if getter(i) else None for i in range(length)]

    if kind == "int_or_vid_list":
        length = getattr(node, f"{accessor_name}Length")()
        getter = getattr(node, accessor_name)
        items = []
        for i in range(length):
            iov = getter(i)
            if iov is None:
                items.append(None)
            elif iov.IsVid():
                v = iov.Vid()
                items.append({"vid": v.Idx()} if v else None)
            else:
                items.append({"literal": iov.Literal()})
        return items

    if kind == "string":
        val = getattr(node, accessor_name)()
        return val.decode("utf-8") if val else None

    # scalar (default)
    return getattr(node, accessor_name)()


def parse_mlx_payload(payload: bytes) -> Dict[str, Any]:
    """Parse raw MLX delegate payload bytes into a dict.

    This is the public entry point for callers that already have the raw
    delegate payload (e.g. from ``extract_delegate_payload``).
    """
    header = MLXHeader.from_bytes(payload)

    if not header.is_valid():
        return {
            "error": f"Invalid MLX magic: {header.magic!r}",
            "header": header.to_dict(),
        }

    fb_data = payload[MLX_HEADER_LENGTH : header.data_segment_offset]
    result: Dict[str, Any] = {
        "header": header.to_dict(),
        "flatbuffer_size": len(fb_data),
        "graph": parse_mlx_flatbuffer(fb_data),
    }

    if header.data_segment_size > 0:
        result["constant_data_size"] = header.data_segment_size

    return result


def parse_executorch_program(pte_data: bytes) -> Dict[str, Any]:  # noqa: C901
    result: Dict[str, Any] = {}

    if len(pte_data) < 8:
        raise ValueError("File too small to be a valid PTE file")

    fb_magic = pte_data[4:8]
    result["flatbuffer_magic"] = fb_magic.decode("utf-8", errors="replace")

    extended_header_offset = 8
    if len(pte_data) > extended_header_offset + 32:
        try:
            header = _ExtendedHeader.from_bytes(pte_data[extended_header_offset:])
            if header.is_valid():
                result["extended_header"] = {
                    "magic": header.magic.decode("utf-8", errors="replace"),
                    "length": header.length,
                    "program_size": header.program_size,
                    "segment_base_offset": header.segment_base_offset,
                    "segment_data_size": header.segment_data_size,
                }
                fb_start = extended_header_offset + header.length
                result["flatbuffer_offset"] = fb_start
                result["flatbuffer_size"] = header.program_size
                result["segment_offset"] = header.segment_base_offset
                result["segment_size"] = header.segment_data_size
        except Exception as e:
            result["header_parse_error"] = str(e)

    try:
        from executorch.exir._serialize._flatbuffer import _program_flatbuffer_to_json

        program_data = json.loads(_program_flatbuffer_to_json(pte_data))
        result["program"] = program_data

        if "execution_plan" in program_data:
            delegates = []
            for plan in program_data["execution_plan"]:
                if "delegates" in plan:
                    for delegate in plan["delegates"]:
                        delegate_info: Dict[str, Any] = {
                            "id": delegate.get("id"),
                            "processed_type": delegate.get("processed", {}).get(
                                "location"
                            ),
                        }
                        processed = delegate.get("processed", {})
                        if "data" in processed:
                            delegate_info["inline_data_size"] = len(processed["data"])
                        if "location" in processed:
                            delegate_info["location"] = processed["location"]
                        delegates.append(delegate_info)
            result["delegates"] = delegates

    except ImportError:
        result["program_parse_error"] = "ExecuTorch FlatBuffer parsing not available"
    except Exception as e:
        result["program_parse_error"] = str(e)

    return result


def _slot_type_display(slot_type: int, style: str = "full") -> str:
    """Return display string for a slot type.

    *style* controls the format:
      - ``"full"``:  "Tensor", "Int", etc.  (for summary tables)
      - ``"short"``: "tid", "vid"            (for instruction I/O lists)
    """
    if style == "short":
        return "tid" if slot_type == 0 else "vid"
    return _SLOT_TYPE_NAMES.get(slot_type, "Unknown")


def _print_slot_map(label: str, slots: List[Dict]) -> None:
    """Print a list of slot-variant dicts with their type names."""
    if not slots:
        return
    print(f"\n  {label}:")
    for i, slot in enumerate(slots):
        type_name = _slot_type_display(slot.get("slot_type", 0))
        print(f"    [{i}]: idx={slot.get('idx')}, type={type_name}")


def show_mlx_summary(pte_data: bytes) -> None:  # noqa: C901
    try:
        mlx_delegates = _find_mlx_delegates(pte_data)
        if not mlx_delegates:
            print("No MLX delegates found in this PTE file.")
            return

        print(f"\n{'='*70}")
        print("MLX DELEGATE SUMMARY")
        print(f"{'='*70}")
        print(f"File contains {len(mlx_delegates)} MLX delegate(s)\n")

        for idx, (delegate_idx, delegate) in enumerate(mlx_delegates):
            print(f"\n--- Delegate {idx} (plan index {delegate_idx}) ---")
            print(f"ID: {delegate.get('id', 'unknown')}")

            try:
                mlx = _load_mlx_payload(pte_data, delegate_index=idx)
            except ValueError as e:
                print(f"  {e}")
                continue

            graph_info = parse_mlx_flatbuffer(mlx.fb_data)

            print("\nMLX Graph Info:")
            for key in (
                "num_constant_tensors",
                "num_input_tensors",
                "num_output_tensors",
                "num_mutable_buffer_tensors",
                "num_temp_tensors",
                "num_values",
                "num_instruction_chains",
            ):
                label = f"  {key + ':':<29}"
                print(f"{label}{graph_info.get(key, '?')}")

            main_idx = graph_info.get("main_chain_idx", 0)
            chains = graph_info.get("instruction_chains", [])
            main_num = "?"
            if main_idx < len(chains):
                main_num = len(chains[main_idx].get("instructions", []))
            print(f"  {'main_chain_idx:':<29}{main_idx} ({main_num} instructions)")
            print(f"  {'init_chain_idx:':<29}{graph_info.get('init_chain_idx', '?')}")

            print("\nI/O Maps:")
            print(
                f"  {'input_map length:':<29}{graph_info.get('input_map_length', '?')}"
            )
            print(
                f"  {'output_map length:':<29}{graph_info.get('output_map_length', '?')}"
            )
            print(
                f"  {'mutable_buffer_map length:':<29}{graph_info.get('mutable_buffer_map_length', '?')}"
            )

            input_len = graph_info.get("input_map_length", 0)
            mutable_len = graph_info.get("mutable_buffer_map_length", 0)
            if input_len and mutable_len is not None:
                print(
                    f"  => regular inputs expected: {input_len - mutable_len} (input_map - mutable_buffer_map)"
                )

            _print_slot_map("Input Map Details", graph_info.get("input_map", []))
            if graph_info.get("mutable_buffer_map"):
                _print_slot_map(
                    "Mutable Buffer Map Details",
                    graph_info["mutable_buffer_map"],
                )
            _print_slot_map("Output Map Details", graph_info.get("output_map", []))

            if mlx.header.data_segment_size > 0:
                print(f"\n  Constant data size: {mlx.header.data_segment_size:,} bytes")

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"Error showing MLX summary: {e}", file=sys.stderr)
        traceback.print_exc()


def show_mlx_instructions(pte_data: bytes) -> None:  # noqa: C901
    try:
        mlx_delegates = _find_mlx_delegates(pte_data)
        if not mlx_delegates:
            print("No MLX delegates found in this PTE file.", file=sys.stderr)
            sys.exit(1)

        if len(mlx_delegates) > 1:
            print(
                f"Found {len(mlx_delegates)} MLX delegate(s) in PTE file\n",
                file=sys.stderr,
            )

        for idx, (delegate_idx, _delegate) in enumerate(mlx_delegates):
            try:
                mlx = _load_mlx_payload(pte_data, delegate_index=idx)
            except ValueError as e:
                print(f"\nError: {e}", file=sys.stderr)
                continue

            graph = parse_mlx_flatbuffer(mlx.fb_data)
            if "error" in graph:
                print(
                    f"\nError parsing delegate {idx}: {graph['error']}",
                    file=sys.stderr,
                )
                continue

            # Print delegate header
            if len(mlx_delegates) > 1:
                print("\n" + "=" * 70)
                print(f"MLX DELEGATE {idx} (plan index {delegate_idx})")
                print("=" * 70)
            else:
                print("\n" + "=" * 70)
                print("MLX Graph Summary")
                print("=" * 70)

            # Basic info
            print(f"Version: {graph.get('version', 'unknown')}")
            print(f"Constant tensors: {graph.get('num_constant_tensors', 0)}")
            print(f"Input tensors: {graph.get('num_input_tensors', 0)}")
            print(f"Output tensors: {graph.get('num_output_tensors', 0)}")
            print(
                f"Mutable buffer tensors: {graph.get('num_mutable_buffer_tensors', 0)}"
            )
            print(f"Temp tensors: {graph.get('num_temp_tensors', 0)}")
            print(f"Values: {graph.get('num_values', 0)}")
            num_chains = graph.get("num_instruction_chains", 0)
            main_idx = graph.get("main_chain_idx", 0)
            init_idx = graph.get("init_chain_idx", -1)
            print(f"Instruction chains: {num_chains}")
            print(f"Main chain idx: {main_idx}")
            if init_idx >= 0:
                print(f"Init chain idx: {init_idx}")

            constant_seg = graph.get("constant_segment", {})
            if constant_seg:
                print(f"Constant data: {constant_seg.get('size', 0):,} bytes")

            # Instruction chains
            for chain_info in graph.get("instruction_chains", []):
                chain_idx = chain_info.get("chain_index", "?")
                label = ""
                if chain_idx == main_idx:
                    label = " (main)"
                elif chain_idx == init_idx:
                    label = " (init)"
                instructions = chain_info.get("instructions", [])
                print(f"\nChain {chain_idx}{label} ({len(instructions)} instructions):")
                for instr in instructions:
                    op_name = instr.get("op_name", f"op_{instr.get('op_type', '?')}")
                    print(f"  [{instr.get('instr_idx', '?')}] {op_name}")

                    for key, value in instr.items():
                        if key in ("instr_idx", "op_type", "op_name"):
                            continue
                        if isinstance(value, dict):
                            if "tid" in value:
                                print(f"      {key}: tid {value['tid']}")
                            elif "vid" in value:
                                print(f"      {key}: vid {value['vid']}")
                            else:
                                print(f"      {key}: {value}")
                        elif value is not None:
                            print(f"      {key}: {value}")

            # Named slots
            named_slots = graph.get("named_slots", [])
            if named_slots:
                print("\nNamed Slots:")
                for slot in named_slots:
                    slot_type = _slot_type_display(
                        slot.get("slot_type", 0), style="short"
                    )
                    print(
                        f"  [{slot.get('slot_idx', '?')}] {slot.get('name', '?')} ({slot_type})"
                    )

            # Input/Output maps
            input_map = graph.get("input_map", [])
            output_map = graph.get("output_map", [])

            if input_map:
                print("\nInputs:")
                for inp in input_map:
                    slot_type = _slot_type_display(
                        inp.get("slot_type", 0), style="short"
                    )
                    print(f"  {slot_type} {inp.get('idx', '?')}")

            if output_map:
                print("\nOutputs:")
                for out in output_map:
                    slot_type = _slot_type_display(
                        out.get("slot_type", 0), style="short"
                    )
                    print(f"  {slot_type} {out.get('idx', '?')}")

            print("=" * 70 + "\n")

    except Exception as e:
        print(f"Error showing MLX instructions: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Inspect ExecuTorch .pte files and extract data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MLX-Specific Options:
  --mlx-summary        Show high-level summary (tensor counts, I/O maps)
  --mlx-instructions   Show detailed instruction list with operation parameters
                       (use this to verify quantization, inspect ops, etc.)

Examples:
  # Basic PTE file inspection
  python -m executorch.backends.mlx.pte_inspector model.pte

  # Show high-level MLX delegate summary
  python -m executorch.backends.mlx.pte_inspector model.pte --mlx-summary

  # Show detailed MLX instructions (verify quantization, inspect operations)
  python -m executorch.backends.mlx.pte_inspector model.pte --mlx-instructions

  # Extract raw delegate payload to binary file
  python -m executorch.backends.mlx.pte_inspector model.pte \\
      --extract-delegate MLXBackend -o delegate.bin
        """,
    )
    parser.add_argument("pte_file", type=Path, help="Path to the .pte file")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--extract-delegate",
        type=str,
        metavar="ID",
        help="Extract delegate payload by ID (e.g., 'mlx')",
    )
    parser.add_argument(
        "--delegate-index",
        type=int,
        default=None,
        metavar="N",
        help="Index of delegate to extract (0-based). If not specified, extracts first matching delegate.",
    )
    parser.add_argument(
        "--parse-mlx",
        action="store_true",
        help="Parse extracted MLX payload (use with --extract-delegate mlx)",
    )
    parser.add_argument(
        "--mlx-summary",
        action="store_true",
        help="Show summary of all MLX delegates (input/output/mutable buffer counts)",
    )
    parser.add_argument(
        "--mlx-instructions",
        action="store_true",
        help="Show detailed MLX instruction list with operands and quantization details",
    )
    parser.add_argument(
        "--format",
        choices=["json", "summary"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )

    args = parser.parse_args()

    if not args.pte_file.exists():
        print(f"Error: File not found: {args.pte_file}", file=sys.stderr)
        sys.exit(1)

    pte_data = args.pte_file.read_bytes()
    print(f"Loaded {len(pte_data)} bytes from {args.pte_file}", file=sys.stderr)

    if args.mlx_instructions:
        show_mlx_instructions(pte_data)
        return

    if args.mlx_summary:
        show_mlx_summary(pte_data)
        return

    if args.extract_delegate:
        payload = extract_delegate_payload(
            pte_data, args.extract_delegate, delegate_index=args.delegate_index
        )
        if payload is None:
            print(
                f"Error: Delegate '{args.extract_delegate}' not found", file=sys.stderr
            )
            sys.exit(1)

        if args.parse_mlx and args.extract_delegate.lower() == "mlx":
            result = parse_mlx_payload(payload)

            output = json.dumps(result, indent=args.indent, default=str)

            if args.output:
                args.output.write_text(output)
                print(f"Wrote parsed MLX data to {args.output}", file=sys.stderr)
            else:
                print(output)
        else:
            if args.output:
                args.output.write_bytes(payload)
                print(f"Wrote {len(payload)} bytes to {args.output}", file=sys.stderr)
            else:
                print(f"Delegate payload: {len(payload)} bytes", file=sys.stderr)
                if len(payload) >= MLX_HEADER_LENGTH:
                    header = MLXHeader.from_bytes(payload)
                    print(f"  Magic: {header.magic!r}", file=sys.stderr)
                    print(
                        f"  Data offset: {header.data_segment_offset}", file=sys.stderr
                    )
                    print(f"  Data size: {header.data_segment_size}", file=sys.stderr)
        return

    result = parse_executorch_program(pte_data)
    result["file_size"] = len(pte_data)
    result["file_path"] = str(args.pte_file)

    if args.format == "summary":
        print(f"PTE File: {args.pte_file}")
        print(f"  Size: {len(pte_data):,} bytes")
        if "extended_header" in result:
            h = result["extended_header"]
            print(f"  Program size: {h['program_size']:,} bytes")
            print(f"  Segment offset: {h['segment_base_offset']:,}")
            print(f"  Segment size: {h['segment_data_size']:,} bytes")
        if "delegates" in result:
            print(f"  Delegates: {len(result['delegates'])}")
            for d in result["delegates"]:
                print(f"    - {d.get('id', 'unknown')}")
    else:
        output = json.dumps(result, indent=args.indent, default=str)

        if args.output:
            args.output.write_text(output)
            print(f"Wrote JSON to {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == "__main__":
    main()
