"""Trace the Cortex-M compilation pipeline pass-by-pass, capturing graph snapshots.

Runs quantization, export, to_edge, then each CortexMPassManager pass individually,
saving a JSON file with per-pass graph snapshots for use with visualize_graph.py.

Usage:
    python3 trace_cortex_m_passes.py --model mobilenet_v2 -o mv2_trace.json

Authored with Claude.
"""

import argparse
import inspect
import json
import os
import sys
import traceback

import torch
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.pass_base import ExportPass
from executorch.exir.program._program import _transform
from torch.export import export
from torchao.quantization.pt2e.export_utils import model_is_exported
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


CATEGORY_COLORS = {
    "backend": "#4caf50",
    "aten_compute": "#2196f3",
    "quantize": "#ff9800",
    "memory": "#9e9e9e",
    "placeholder": "#03a9f4",
    "param": "#78909c",
    "delegate": "#ab47bc",
}


def categorize_node(op_name: str) -> str:
    name = op_name.lower()
    if "cortex_m" in name:
        return "backend"
    if any(
        k in name
        for k in (
            "quantize_per_tensor",
            "dequantize_per_",
            "quantize_per_channel",
            "dequantize_per_channel",
        )
    ):
        return "quantize"
    if any(
        k in name
        for k in (
            "view",
            "clone",
            "permute",
            "slice",
            "copy",
            "expand",
            "reshape",
            "t_copy",
            "unsqueeze",
            "squeeze",
        )
    ):
        return "memory"
    if any(k in name for k in ("placeholder", "output", "getitem", "get_attr")):
        return "placeholder"
    if "delegate" in name:
        return "delegate"
    return "aten_compute"


def _make_label(op_name: str) -> str:
    name = op_name.split("::")[-1] if "::" in op_name else op_name
    if "." in name:
        name = name.rsplit(".", 1)[0]
    if len(name) > 30:
        name = name[:27] + "..."
    return name


def _serialize_qparams(qparams_dict):
    """Serialize a dict[int, QuantArgs] to JSON-safe form."""
    if not qparams_dict:
        return None
    result = {}
    for idx, qa in qparams_dict.items():
        result[str(idx)] = {
            "scale": qa.scale,
            "zp": qa.zp,
            "qmin": qa.qmin,
            "qmax": qa.qmax,
            "dtype": str(qa.dtype),
            "axis": qa.axis,
            "per_channel": qa.per_channel,
        }
    return result


def detect_qdq_groups(graph) -> dict:
    """Detect DQ -> compute_op -> Q chains and assign group IDs.

    Returns {node_name: group_id} for nodes in DQ->op->Q chains.
    """
    groups = {}
    group_id = 0

    q_op_set = set()
    dq_op_set = set()
    for node in graph.nodes:
        if node.op == "call_function":
            target = node.target
            if target in Q_OPS:
                q_op_set.add(target)
            if target in DQ_OPS:
                dq_op_set.add(target)

    for node in graph.nodes:
        if node.op != "call_function" or node.target not in Q_OPS:
            continue
        # This is a Q node. Find its compute input.
        if not node.args:
            continue
        compute_node = node.args[0]
        if not hasattr(compute_node, "name"):
            continue
        if not hasattr(compute_node, "target"):
            continue
        # Skip if compute_node is itself a Q/DQ op
        if compute_node.target in Q_OPS or compute_node.target in DQ_OPS:
            continue

        # Find DQ nodes feeding the compute node
        dq_nodes = []
        for arg in compute_node.args:
            if hasattr(arg, "target") and arg.target in DQ_OPS:
                dq_nodes.append(arg)
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if hasattr(a, "target") and a.target in DQ_OPS:
                        dq_nodes.append(a)

        if not dq_nodes:
            continue

        # Assign all members the same group_id
        members = [n.name for n in dq_nodes] + [compute_node.name, node.name]
        for m in members:
            if m not in groups:
                groups[m] = group_id
        group_id += 1

    return groups


def extract_from_exported_program(ep, stage_name, qdq_groups=None):
    """Walk an ExportedProgram's graph and extract visualization data."""
    graph = ep.graph

    nodes = []
    edges = []
    node_map = {}

    for node in graph.nodes:
        node_id = node.name
        op_name = node.op
        if node.op == "call_function":
            op_name = str(node.target)
        elif node.op == "call_method":
            op_name = node.target

        details = {"op": node.op, "target": str(getattr(node, "target", ""))}

        # Shape/dtype from meta
        meta_val = node.meta.get("val")
        if meta_val is not None:
            if hasattr(meta_val, "shape"):
                details["shape"] = str(list(meta_val.shape))
                details["dtype"] = str(meta_val.dtype)
            elif isinstance(meta_val, (list, tuple)):
                shapes = []
                for v in meta_val:
                    if hasattr(v, "shape"):
                        shapes.append(f"{list(v.shape)} {v.dtype}")
                if shapes:
                    details["shapes"] = shapes

        # Stack trace
        stack_trace = node.meta.get("stack_trace")
        if stack_trace:
            # Truncate long traces to last 500 chars
            if len(stack_trace) > 500:
                stack_trace = "..." + stack_trace[-500:]
            details["stack_trace"] = stack_trace

        # Quantization params (post-fold stages)
        input_qparams = node.meta.get("input_qparams")
        if input_qparams:
            serialized = _serialize_qparams(input_qparams)
            if serialized:
                details["input_qparams"] = serialized

        output_qparams = node.meta.get("output_qparams")
        if output_qparams:
            serialized = _serialize_qparams(output_qparams)
            if serialized:
                details["output_qparams"] = serialized

        category = categorize_node(op_name)
        label = _make_label(op_name)

        # Meaningful labels for placeholders
        if node.op == "placeholder":
            target = str(getattr(node, "target", ""))
            if target.startswith("p_") and target.endswith("_weight"):
                label = "weight"
                category = "param"
            elif target.startswith("p_") and target.endswith("_bias"):
                label = "bias"
                category = "param"
            elif target.startswith("b_") and "running_mean" in target:
                label = "bn_mean"
                category = "param"
            elif target.startswith("b_") and "running_var" in target:
                label = "bn_var"
                category = "param"
            elif target == "x" or not target.startswith(("p_", "b_")):
                label = "input"
        elif node.op == "output":
            label = "output"

        node_data = {
            "id": node_id,
            "label": label,
            "w": max(len(label) * 8 + 16, 60),
            "category": category,
            "op_name": op_name,
            "details": details,
        }

        if qdq_groups and node_id in qdq_groups:
            node_data["qdq_group_id"] = qdq_groups[node_id]

        node_map[node_id] = len(nodes)
        nodes.append(node_data)

        for arg in node.args:
            if hasattr(arg, "name") and arg.name in node_map:
                edges.append({"source": arg.name, "target": node_id})
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if hasattr(a, "name") and a.name in node_map:
                        edges.append({"source": a.name, "target": node_id})

    category_counts = {}
    for n in nodes:
        cat = n["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "metadata": {
            "model_name": stage_name,
            "source_type": "trace",
            "total_nodes": len(nodes),
            "category_counts": category_counts,
            "error": None,
        },
        "nodes": nodes,
        "edges": edges,
    }


def _to_channels_last(x):
    if isinstance(x, torch.Tensor):
        return x.to(memory_format=torch.channels_last) if x.dim() == 4 else x
    elif isinstance(x, tuple):
        return tuple(_to_channels_last(t) for t in x)
    return x


def get_model(model_name: str):
    """Load a model by name, returning (module, example_inputs)."""
    if model_name == "mobilenet_v2":
        from torchvision.models import mobilenet_v2

        model = mobilenet_v2(weights=None)
        model.eval()
        return model, (torch.randn(1, 3, 224, 224),)
    elif model_name == "lstm":
        from torch.nn.quantizable.modules import rnn

        model = rnn.LSTM(10, 20, 2)
        model.eval()
        example_inputs = (
            torch.randn(5, 3, 10),
            (torch.randn(2, 3, 20), torch.randn(2, 3, 20)),
        )
        return model, example_inputs
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_pipeline(model_name: str) -> dict:
    """Run the full Cortex-M pipeline, capturing snapshots after each stage."""
    model, example_inputs = get_model(model_name)
    snapshots = []

    def _randn_like(x):
        if isinstance(x, torch.Tensor):
            return torch.randn_like(x)
        elif isinstance(x, tuple):
            return tuple(_randn_like(t) for t in x)
        return x

    # --- Stage 1: Quantize ---
    print("Quantizing...")
    quantizer = CortexMQuantizer()
    model = torch.export.export_for_training(model, example_inputs).module()
    prepared = prepare_pt2e(model, quantizer)
    # Calibrate with random data
    with torch.no_grad():
        for _ in range(5):
            prepared(*[_randn_like(t) for t in example_inputs])
    quantized = convert_pt2e(prepared)

    # --- Stage 2: Export ---
    print("Exporting...")
    ep = export(quantized, example_inputs, strict=True)
    qdq_groups = detect_qdq_groups(ep.graph)
    snapshots.append(
        extract_from_exported_program(ep, "1_post_export", qdq_groups)
    )
    print(f"  1_post_export: {snapshots[-1]['metadata']['total_nodes']} nodes")

    # --- Stage 3: to_edge ---
    print("Converting to edge...")
    edge_config = EdgeCompileConfig(
        preserve_ops=[
            torch.ops.aten.linear.default,
            torch.ops.aten.hardsigmoid.default,
            torch.ops.aten.hardsigmoid_.default,
            torch.ops.aten.hardswish.default,
            torch.ops.aten.hardswish_.default,
        ],
        _check_ir_validity=False,
        _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
    )
    edge_program = to_edge(ep, compile_config=edge_config)
    edge_ep = edge_program.exported_program()
    qdq_groups = detect_qdq_groups(edge_ep.graph)
    snapshots.append(
        extract_from_exported_program(edge_ep, "2_post_to_edge", qdq_groups)
    )
    print(f"  2_post_to_edge: {snapshots[-1]['metadata']['total_nodes']} nodes")

    # --- Stages 4-10+: Individual passes ---
    pass_list = CortexMPassManager.pass_list
    ep = edge_ep

    for i, pass_cls in enumerate(pass_list):
        pass_name = f"{i + 3}_{pass_cls.__name__}"
        print(f"Running {pass_name}...")

        try:
            signature = inspect.signature(pass_cls.__init__)
            if "exported_program" in signature.parameters:
                transform_pass = pass_cls(ep)
            elif issubclass(pass_cls, ExportPass):
                transform_pass = pass_cls()
            else:
                raise RuntimeError(
                    f"Unexpected pass type: {pass_cls} ({type(pass_cls)})"
                )
            ep = _transform(ep, transform_pass)

            # Detect QDQ groups for pre-fold stages
            qdq_groups_for_pass = None
            if pass_cls != FoldAndAnnotateQParamsPass:
                try:
                    qdq_groups_for_pass = detect_qdq_groups(ep.graph)
                except Exception:
                    pass

            snapshot = extract_from_exported_program(
                ep, pass_name, qdq_groups_for_pass
            )
            snapshots.append(snapshot)
            print(f"  {pass_name}: {snapshot['metadata']['total_nodes']} nodes")

        except Exception as exc:
            print(f"  ERROR in {pass_name}: {exc}", file=sys.stderr)
            error_snapshot = extract_from_exported_program(
                ep, f"{pass_name}_ERROR"
            )
            error_snapshot["metadata"]["error"] = {
                "pass_name": pass_name,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            snapshots.append(error_snapshot)
            break

    return {"model_name": model_name, "passes": snapshots}


# Import here so it's available for the isinstance check in the pass loop
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Trace Cortex-M compilation passes and output JSON snapshots"
    )
    parser.add_argument(
        "--model",
        default="mobilenet_v2",
        help="Model name (default: mobilenet_v2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <model>_trace.json)",
    )
    args = parser.parse_args()

    output = args.output or f"{args.model}_trace.json"
    result = run_pipeline(args.model)

    with open(output, "w") as f:
        json.dump(result, f)
    print(
        f"Wrote {output} ({len(result['passes'])} passes, "
        f"{os.path.getsize(output) / 1024:.0f} KB)"
    )


if __name__ == "__main__":
    main()
