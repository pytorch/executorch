#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
"""Trace any ExecuTorch backend's compilation pipeline pass-by-pass.

Runs quantization, export, to_edge, then each backend pass individually,
saving a JSON file with per-pass graph snapshots for visualization with
html_visualization.py.

Usage:
    # Cortex-M backend:
    python -m executorch.devtools.visualization.trace_passes \\
        --backend cortex_m --model mobilenet_v2 -o trace.json

    # XNNPACK backend:
    python -m executorch.devtools.visualization.trace_passes \\
        --backend xnnpack --model mobilenet_v2 -o trace.json

    # Cadence backend:
    python -m executorch.devtools.visualization.trace_passes \\
        --backend cadence --model mobilenet_v2 -o trace.json

    # Skip quantization (trace passes on a float model):
    python -m executorch.devtools.visualization.trace_passes \\
        --backend xnnpack --model mobilenet_v2 --no-quantize -o trace.json

    # Then visualize:
    python -m executorch.devtools.visualization.html_visualization trace.json -o trace.html

Authored with Claude.
"""

import argparse
import inspect
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.pass_base import ExportPass
from executorch.exir.program._program import _transform
from torch.export import export


# ---------------------------------------------------------------------------
# Node categorization (shared with html_visualization.py)
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "backend": "#4caf50",
    "aten_compute": "#2196f3",
    "quantize": "#ff9800",
    "memory": "#9e9e9e",
    "placeholder": "#03a9f4",
    "param": "#78909c",
    "delegate": "#ab47bc",
}

_BACKEND_OP_PREFIXES = (
    "cortex_m",
    "cadence",
    "qaisw",
)


def categorize_node(op_name: str) -> str:
    name = op_name.lower()
    if any(prefix in name for prefix in _BACKEND_OP_PREFIXES):
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


# ---------------------------------------------------------------------------
# QDQ group detection
# ---------------------------------------------------------------------------


def _get_qdq_ops():
    """Get Q/DQ op sets. Works with or without Arm constants."""
    q_ops = set()
    dq_ops = set()
    try:
        from executorch.backends.arm.constants import DQ_OPS, Q_OPS

        q_ops.update(Q_OPS)
        dq_ops.update(DQ_OPS)
    except ImportError:
        pass
    # Fallback: common quantize/dequantize ops
    for op_name in [
        "quantize_per_tensor",
        "quantize_per_channel",
        "quantize_per_tensor_tensor",
    ]:
        op = getattr(torch.ops.quantized_decomposed, op_name, None)
        if op is not None:
            q_ops.add(op.default)
    for op_name in [
        "dequantize_per_tensor",
        "dequantize_per_channel",
        "dequantize_per_tensor_tensor",
    ]:
        op = getattr(torch.ops.quantized_decomposed, op_name, None)
        if op is not None:
            dq_ops.add(op.default)
    return q_ops, dq_ops


def detect_qdq_groups(graph) -> dict:
    """Detect DQ -> compute_op -> Q chains and assign group IDs."""
    q_ops, dq_ops = _get_qdq_ops()
    if not q_ops or not dq_ops:
        return {}

    groups = {}
    group_id = 0

    for node in graph.nodes:
        if node.op != "call_function" or node.target not in q_ops:
            continue
        if not node.args:
            continue
        compute_node = node.args[0]
        if not hasattr(compute_node, "name") or not hasattr(compute_node, "target"):
            continue
        if compute_node.target in q_ops or compute_node.target in dq_ops:
            continue

        dq_nodes = []
        for arg in compute_node.args:
            if hasattr(arg, "target") and arg.target in dq_ops:
                dq_nodes.append(arg)
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if hasattr(a, "target") and a.target in dq_ops:
                        dq_nodes.append(a)

        if not dq_nodes:
            continue

        members = [n.name for n in dq_nodes] + [compute_node.name, node.name]
        for m in members:
            if m not in groups:
                groups[m] = group_id
        group_id += 1

    return groups


# ---------------------------------------------------------------------------
# Graph snapshot extraction
# ---------------------------------------------------------------------------


def _serialize_qparams(qparams_dict):
    if not qparams_dict:
        return None
    result = {}
    for idx, qa in qparams_dict.items():
        entry = {}
        for attr in ("scale", "zp", "qmin", "qmax", "axis", "per_channel"):
            val = getattr(qa, attr, None)
            if val is not None:
                entry[attr] = val
        dtype = getattr(qa, "dtype", None)
        if dtype is not None:
            entry["dtype"] = str(dtype)
        result[str(idx)] = entry
    return result


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

        stack_trace = node.meta.get("stack_trace")
        if stack_trace:
            if len(stack_trace) > 500:
                stack_trace = "..." + stack_trace[-500:]
            details["stack_trace"] = stack_trace

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


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


class BackendConfig:
    """Configuration for tracing a backend's compilation pipeline."""

    def __init__(
        self,
        name: str,
        quantizer_cls: Optional[str] = None,
        pass_list_source: Optional[str] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
        # Some passes should skip QDQ detection after them
        skip_qdq_after: Optional[List[str]] = None,
    ):
        self.name = name
        self._quantizer_cls_path = quantizer_cls
        self._pass_list_source = pass_list_source
        self.edge_compile_config = edge_compile_config or EdgeCompileConfig(
            _check_ir_validity=False,
        )
        self.skip_qdq_after = set(skip_qdq_after or [])

    def get_quantizer(self):
        if not self._quantizer_cls_path:
            return None
        module_path, cls_name = self._quantizer_cls_path.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        return cls()

    def get_pass_list(self) -> List[Type]:
        """Return a list of pass classes to iterate through."""
        if not self._pass_list_source:
            return []

        import importlib

        if ":" in self._pass_list_source:
            # "module.path:function_or_attribute" format
            module_path, attr = self._pass_list_source.split(":", 1)
            mod = importlib.import_module(module_path)
            obj = getattr(mod, attr)
            if callable(obj) and not isinstance(obj, type):
                return obj()
            return list(obj)
        else:
            # "module.path.Class.pass_list" — dotted attribute access
            parts = self._pass_list_source.rsplit(".", 1)
            module_path, attr = parts
            # Try importing as module.Class.attr
            try:
                mod = importlib.import_module(module_path)
                return list(getattr(mod, attr))
            except (ImportError, AttributeError):
                # Try one level up: module.Class has attr
                mod_path, cls_name = module_path.rsplit(".", 1)
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)
                return list(getattr(cls, attr))


# Backend configurations — add new backends here
_BACKEND_REGISTRY: Dict[str, BackendConfig] = {}


def register_backend(config: BackendConfig):
    _BACKEND_REGISTRY[config.name] = config


def _register_builtin_backends():
    register_backend(
        BackendConfig(
            name="cortex_m",
            quantizer_cls=(
                "executorch.backends.cortex_m.quantizer.quantizer"
                ".CortexMQuantizer"
            ),
            pass_list_source=(
                "executorch.backends.cortex_m.passes"
                ".cortex_m_pass_manager.CortexMPassManager.pass_list"
            ),
            edge_compile_config=EdgeCompileConfig(
                preserve_ops=[
                    torch.ops.aten.linear.default,
                    torch.ops.aten.hardsigmoid.default,
                    torch.ops.aten.hardsigmoid_.default,
                    torch.ops.aten.hardswish.default,
                    torch.ops.aten.hardswish_.default,
                ],
                _check_ir_validity=False,
                _core_aten_ops_exception_list=[
                    torch.ops.aten.max_pool2d.default,
                ],
            ),
            skip_qdq_after=["FoldAndAnnotateQParamsPass"],
        )
    )

    register_backend(
        BackendConfig(
            name="xnnpack",
            quantizer_cls=(
                "executorch.backends.xnnpack.quantizer"
                ".xnnpack_quantizer.XNNPACKQuantizer"
            ),
            pass_list_source=(
                "executorch.devtools.visualization.trace_passes"
                ":_default_pass_list"
            ),
        )
    )

    register_backend(
        BackendConfig(
            name="cadence",
            quantizer_cls=(
                "executorch.backends.cadence.aot.quantizer"
                ".quantizer.CadenceDefaultQuantizer"
            ),
            pass_list_source=(
                "executorch.backends.cadence.aot.passes"
                ":get_passes_in_default_order"
            ),
        )
    )

    register_backend(
        BackendConfig(
            name="vulkan",
            quantizer_cls=(
                "executorch.backends.vulkan.quantizer"
                ".vulkan_quantizer.VulkanQuantizer"
            ),
            # Vulkan has no static pass list — passes are inline in preprocess()
            pass_list_source=None,
        )
    )

    register_backend(
        BackendConfig(
            name="qnn",
            quantizer_cls=(
                "executorch.backends.qualcomm.quantizer"
                ".quantizer.QnnQuantizer"
            ),
            # QNN uses dynamic pipeline methods, no static pass list
            pass_list_source=None,
        )
    )


# ---------------------------------------------------------------------------
# Pass instantiation
# ---------------------------------------------------------------------------


def _instantiate_pass(pass_cls, exported_program):
    """Instantiate a pass class, passing exported_program if needed."""
    sig = inspect.signature(pass_cls.__init__)
    params = list(sig.parameters.keys())
    # Skip 'self'
    if "exported_program" in params:
        return pass_cls(exported_program)
    elif len(params) > 1 and params[1] == "exported_program":
        return pass_cls(exported_program)
    else:
        try:
            return pass_cls()
        except TypeError:
            # Some passes require arguments we can't infer
            return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def get_model(model_name: str) -> Tuple[torch.nn.Module, tuple]:
    """Load a model by name, returning (module, example_inputs)."""
    if model_name == "mobilenet_v2":
        from torchvision.models import mobilenet_v2

        model = mobilenet_v2(weights=None)
        model.eval()
        return model, (torch.randn(1, 3, 224, 224),)
    elif model_name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        model.eval()
        return model, (torch.randn(1, 3, 224, 224),)
    elif model_name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.eval()
        return model, (torch.randn(1, 3, 224, 224),)
    elif model_name == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.eval()
        return model, (torch.randn(1, 3, 224, 224),)
    elif model_name == "lstm":
        from torch.nn.quantizable.modules import rnn

        model = rnn.LSTM(10, 20, 2)
        model.eval()
        return model, (
            torch.randn(5, 3, 10),
            (torch.randn(2, 3, 20), torch.randn(2, 3, 20)),
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: mobilenet_v2, mobilenet_v3_small, resnet18, resnet50, lstm"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _randn_like(x):
    if isinstance(x, torch.Tensor):
        return torch.randn_like(x)
    elif isinstance(x, tuple):
        return tuple(_randn_like(t) for t in x)
    return x


def run_pipeline(
    backend_config: BackendConfig,
    model_name: str,
    quantize: bool = True,
) -> dict:
    """Run a backend's compilation pipeline, capturing snapshots."""
    model, example_inputs = get_model(model_name)
    snapshots = []
    stage_num = 0

    # --- Stage: Quantize ---
    if quantize:
        quantizer = backend_config.get_quantizer()
        if quantizer is None:
            print(
                f"Warning: no quantizer configured for {backend_config.name}, "
                f"skipping quantization"
            )
        else:
            stage_num += 1
            print("Quantizing...")
            from torchao.quantization.pt2e.quantize_pt2e import (
                convert_pt2e,
                prepare_pt2e,
            )

            model = torch.export.export_for_training(
                model, example_inputs
            ).module()
            prepared = prepare_pt2e(model, quantizer)
            with torch.no_grad():
                for _ in range(5):
                    prepared(*[_randn_like(t) for t in example_inputs])
            model = convert_pt2e(prepared)

    # --- Stage: Export ---
    stage_num += 1
    print("Exporting...")
    ep = export(model, example_inputs, strict=True)
    qdq_groups = detect_qdq_groups(ep.graph)
    snapshots.append(
        extract_from_exported_program(
            ep, f"{stage_num}_post_export", qdq_groups
        )
    )
    print(f"  {stage_num}_post_export: {snapshots[-1]['metadata']['total_nodes']} nodes")

    # --- Stage: to_edge ---
    stage_num += 1
    print("Converting to edge...")
    edge_program = to_edge(ep, compile_config=backend_config.edge_compile_config)
    edge_ep = edge_program.exported_program()
    qdq_groups = detect_qdq_groups(edge_ep.graph)
    snapshots.append(
        extract_from_exported_program(
            edge_ep, f"{stage_num}_post_to_edge", qdq_groups
        )
    )
    print(
        f"  {stage_num}_post_to_edge: "
        f"{snapshots[-1]['metadata']['total_nodes']} nodes"
    )

    # --- Stages: Individual passes ---
    pass_list = backend_config.get_pass_list()
    if not pass_list:
        print(
            f"  No static pass list for {backend_config.name} — "
            f"showing export/edge stages only"
        )
    else:
        ep = edge_ep
        for i, pass_cls in enumerate(pass_list):
            stage_num += 1
            pass_name = f"{stage_num}_{pass_cls.__name__}"
            print(f"Running {pass_name}...")

            try:
                transform_pass = _instantiate_pass(pass_cls, ep)
                if transform_pass is None:
                    print(f"  Skipping {pass_name} (cannot instantiate)")
                    stage_num -= 1
                    continue

                ep = _transform(ep, transform_pass)

                qdq_groups_for_pass = None
                if pass_cls.__name__ not in backend_config.skip_qdq_after:
                    try:
                        qdq_groups_for_pass = detect_qdq_groups(ep.graph)
                    except Exception:
                        pass

                snapshot = extract_from_exported_program(
                    ep, pass_name, qdq_groups_for_pass
                )
                snapshots.append(snapshot)
                print(
                    f"  {pass_name}: "
                    f"{snapshot['metadata']['total_nodes']} nodes"
                )

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

    return {
        "model_name": f"{model_name} ({backend_config.name})",
        "passes": snapshots,
    }


# ---------------------------------------------------------------------------
# XNNPACK helper — extract default pass list from XNNPACKPassManager
# ---------------------------------------------------------------------------


def _default_pass_list():
    """Extract the default pass list from XNNPACKPassManager.

    XNNPACKPassManager stores passes as an instance attribute, so we need
    to peek at the default set in __init__.
    """
    try:
        from executorch.backends.xnnpack._passes import XNNPACKPassManager

        # XNNPACKPassManager.__init__ takes exported_program, but we just
        # need the default pass class list. Use the source to get it.
        src = inspect.getsource(XNNPACKPassManager.__init__)
        # The pass list is assigned as self.passes = [...] in __init__
        # Rather than parsing source, just grab from a dummy instance
        # Actually, we can't instantiate without an exported_program.
        # Fall back to the known default list.
        from executorch.backends.xnnpack._passes import (
            ChannelsLastTaggedReshapePass,
            ConstPropPass,
            Conv1dUnsqueezePass,
            ConvertToLinearPass,
            ConvertToSDPAPass,
            ConvertToUpsampleBilinear2d,
            DecomposeBatchNorm,
            DecomposeConcatenate,
            DimOrderOpsRevertPass,
            FuseActivationPass,
            FuseBatchNormPass,
            PReLUReshapePass,
            PropagateCustomMetaPass,
            RemoveGetItemPass,
            RemoveRedundantCopyPass,
            XNNPACKRemoveCloneOpsTransform,
        )

        return [
            XNNPACKRemoveCloneOpsTransform,
            DimOrderOpsRevertPass,
            ConvertToUpsampleBilinear2d,
            ConvertToLinearPass,
            PropagateCustomMetaPass,
            ConvertToSDPAPass,
            ConstPropPass,
            FuseBatchNormPass,
            DecomposeBatchNorm,
            FuseActivationPass,
            DecomposeConcatenate,
            RemoveGetItemPass,
            Conv1dUnsqueezePass,
            PReLUReshapePass,
            ChannelsLastTaggedReshapePass,
            RemoveRedundantCopyPass,
        ]
    except ImportError as e:
        print(f"Warning: Could not import XNNPACK passes: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    _register_builtin_backends()

    parser = argparse.ArgumentParser(
        description="Trace ExecuTorch backend compilation passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available backends: "
            + ", ".join(sorted(_BACKEND_REGISTRY.keys()))
            + "\n\nAvailable models: mobilenet_v2, mobilenet_v3_small, "
            "resnet18, resnet50, lstm"
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend name (e.g., cortex_m, xnnpack, cadence)",
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
        help="Output JSON path (default: <model>_<backend>_trace.json)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization (trace on float model)",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit",
    )
    args = parser.parse_args()

    if args.list_backends:
        print("Available backends:")
        for name, cfg in sorted(_BACKEND_REGISTRY.items()):
            has_passes = "with passes" if cfg._pass_list_source else "export/edge only"
            has_quant = "quantized" if cfg._quantizer_cls_path else "float"
            print(f"  {name:15s} ({has_quant}, {has_passes})")
        return

    if not args.backend:
        parser.error("--backend is required (use --list-backends to see options)")

    if args.backend not in _BACKEND_REGISTRY:
        print(
            f"Error: unknown backend '{args.backend}'. "
            f"Available: {', '.join(sorted(_BACKEND_REGISTRY.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    backend_config = _BACKEND_REGISTRY[args.backend]
    output = args.output or f"{args.model}_{args.backend}_trace.json"

    result = run_pipeline(
        backend_config,
        args.model,
        quantize=not args.no_quantize,
    )

    with open(output, "w") as f:
        json.dump(result, f)
    print(
        f"\nWrote {output} ({len(result['passes'])} passes, "
        f"{os.path.getsize(output) / 1024:.0f} KB)"
    )
    print(
        f"Visualize: python -m executorch.devtools.visualization"
        f".html_visualization {output} -o {output.replace('.json', '.html')}"
    )


if __name__ == "__main__":
    main()
