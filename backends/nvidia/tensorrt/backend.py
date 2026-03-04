# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT backend implementation for ExecuTorch."""

import logging
from typing import Any, Dict, final, List, Optional, Tuple

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram

from executorch.backends.nvidia.tensorrt.compile_spec import (
    TensorRTCompileSpec,
    TensorRTPrecision,
)
from executorch.backends.nvidia.tensorrt.converter_registry import (
    lookup_converter,
    needs_edge_program,
)
from executorch.backends.nvidia.tensorrt.serialization import (
    serialize_blob,
    TensorRTBlobMetadata,
    TensorRTIOBinding,
)
from executorch.backends.nvidia.tensorrt.converters import (
    clear_converter_weight_storage,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@final
class TensorRTBackend(BackendDetails):
    """TensorRT backend for accelerating models on NVIDIA GPUs.

    This backend compiles ExecuTorch edge programs to TensorRT engines
    for optimized inference on NVIDIA hardware.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Compile edge program to TensorRT engine.

        Args:
            edge_program: The edge dialect program to compile.
            compile_specs: Backend-specific compilation options.

        Returns:
            PreprocessResult containing the serialized TensorRT engine.
        """
        try:
            import tensorrt as trt
        except ImportError as e:
            raise RuntimeError(
                "TensorRT is not available. Please install TensorRT to use this backend."
            ) from e

        # Import converters to trigger registration
        from executorch.backends.nvidia.tensorrt import (  # noqa: F401
            converters as _converters,
        )
        from executorch.backends.nvidia.tensorrt.converter_utils import (
            ConversionContext,
            get_op_name,
            get_trt_tensor,
            torch_dtype_to_trt,
        )

        # Parse compile specs
        spec = TensorRTCompileSpec.from_compile_specs(compile_specs)
        if spec is None:
            spec = TensorRTCompileSpec()

        graph_module = edge_program.graph_module

        # Identify input and output nodes
        input_nodes = _get_input_nodes(graph_module, edge_program)
        output_nodes = _get_output_nodes(graph_module)

        # Create TensorRT builder and network
        trt_logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        if network is None:
            raise RuntimeError("Failed to create TensorRT network")

        # Create conversion context for this build
        ctx = ConversionContext(net=network)

        # Build the network
        input_map = _add_network_inputs(network, input_nodes, torch_dtype_to_trt)
        # Add params/buffers as constant tensors
        _add_params_to_input_map(
            graph_module, edge_program, network, input_map, get_trt_tensor
        )
        _process_graph_nodes(
            graph_module, edge_program, network, input_map, get_trt_tensor, get_op_name, ctx
        )
        _mark_network_outputs(network, output_nodes, input_map)

        # Collect I/O bindings from network
        io_bindings = _collect_io_bindings(network)

        # Configure and build engine
        config = _create_builder_config(builder, spec, trt)
        _add_optimization_profile(
            builder, config, network, input_nodes, edge_program, trt
        )
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Serialize with metadata
        metadata = TensorRTBlobMetadata(io_bindings=io_bindings)
        blob = serialize_blob(bytes(serialized_engine), metadata)

        return PreprocessResult(processed_bytes=blob)


def _get_input_nodes(
    graph_module: torch.fx.GraphModule,
    exported_program: ExportedProgram,
) -> List[torch.fx.Node]:
    """Get graph input placeholder nodes (excluding parameters/buffers)."""
    input_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if not _is_param_or_buffer(node, exported_program):
                input_nodes.append(node)
    return input_nodes


def _get_output_nodes(graph_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    """Get nodes that are graph outputs."""
    output_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == "output":
            for arg in node.args:
                if isinstance(arg, (list, tuple)):
                    output_nodes.extend(
                        item for item in arg if isinstance(item, torch.fx.Node)
                    )
                elif isinstance(arg, torch.fx.Node):
                    output_nodes.append(arg)
    return output_nodes


def _is_param_or_buffer(
    node: torch.fx.Node, exported_program: ExportedProgram
) -> bool:
    """Check if a placeholder node is a parameter, buffer, or lifted constant."""
    if node.op != "placeholder":
        return False

    if hasattr(exported_program, "state_dict"):
        if node.name in exported_program.state_dict:
            return True

    if hasattr(exported_program, "graph_signature"):
        sig = exported_program.graph_signature
        if hasattr(sig, "inputs_to_parameters"):
            if node.name in sig.inputs_to_parameters:
                return True
        if hasattr(sig, "inputs_to_buffers"):
            if node.name in sig.inputs_to_buffers:
                return True
        # Lifted constants (e.g., c_enc_lifted_tensor_0) are constant
        # data embedded in the ExportedProgram.  They should be baked
        # into the TRT engine as constants, not passed as runtime inputs.
        if hasattr(sig, "inputs_to_lifted_tensor_constants"):
            if node.name in sig.inputs_to_lifted_tensor_constants:
                return True
        if hasattr(sig, "inputs_to_lifted_custom_objs"):
            if node.name in sig.inputs_to_lifted_custom_objs:
                return True

    # Also check constants dict
    if hasattr(exported_program, "constants"):
        if node.name in exported_program.constants:
            return True

    return False


def _add_params_to_input_map(
    graph_module: torch.fx.GraphModule,
    exported_program: ExportedProgram,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    get_trt_tensor_fn: Any,
) -> None:
    """Add parameters and buffers as constant TensorRT tensors to input_map.

    In ExecuTorch's edge dialect, parameters are often "lifted" as placeholder
    inputs rather than get_attr nodes. This function identifies these placeholder
    nodes that represent parameters/buffers and adds them to input_map as
    TensorRT constant tensors.
    """
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            # Skip if already in input_map (it's a real input, not a param)
            if node in input_map:
                continue

            param_tensor = None

            # Try to get from state_dict first
            if hasattr(exported_program, "state_dict"):
                if node.name in exported_program.state_dict:
                    param_tensor = exported_program.state_dict[node.name]

            # Try to get from graph_signature mapping
            if param_tensor is None and hasattr(exported_program, "graph_signature"):
                sig = exported_program.graph_signature
                param_name = None
                if hasattr(sig, "inputs_to_parameters"):
                    param_name = sig.inputs_to_parameters.get(node.name)
                if param_name is None and hasattr(sig, "inputs_to_buffers"):
                    param_name = sig.inputs_to_buffers.get(node.name)

                if param_name is not None and hasattr(exported_program, "state_dict"):
                    param_tensor = exported_program.state_dict.get(param_name)

            # Try lifted tensor constants (e.g., c_enc_lifted_tensor_0).
            # These are constant data embedded in the ExportedProgram that
            # should be baked into the TRT engine, not passed at runtime.
            if param_tensor is None and hasattr(exported_program, "graph_signature"):
                sig = exported_program.graph_signature
                if hasattr(sig, "inputs_to_lifted_tensor_constants"):
                    const_fqn = sig.inputs_to_lifted_tensor_constants.get(node.name)
                    if const_fqn is not None and hasattr(exported_program, "constants"):
                        param_tensor = exported_program.constants.get(const_fqn)

            # If we found a parameter/buffer/constant tensor, add it
            if param_tensor is not None:
                if isinstance(param_tensor, torch.nn.Parameter):
                    param_tensor = param_tensor.data
                if isinstance(param_tensor, torch.Tensor):
                    # get_trt_tensor handles dtype conversion (int64→int32, float64→float32)
                    # via create_constant in converter_utils.py
                    input_map[node] = get_trt_tensor_fn(
                        network, param_tensor, f"param_{node.name}"
                    )


def _get_tensor_shape_and_dtype(
    node: torch.fx.Node,
) -> Tuple[Optional[Tuple[int, ...]], Optional[torch.dtype]]:
    """Extract tensor shape and dtype from node metadata."""
    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            return tuple(val.shape), val.dtype
        if hasattr(val, "shape") and hasattr(val, "dtype"):
            return tuple(val.shape), val.dtype
    return None, None


def _get_attr_value(
    graph_module: torch.fx.GraphModule, attr_name: str
) -> Optional[torch.Tensor]:
    """Get attribute value from graph module."""
    try:
        parts = attr_name.split(".")
        obj = graph_module
        for part in parts:
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, torch.nn.Parameter):
            return obj.data
        return None
    except AttributeError:
        return None


def _is_symint(val: Any) -> bool:
    """Check if a value is a SymInt (symbolic integer from dynamic shapes)."""
    return hasattr(val, "node") and type(val).__name__ == "SymInt"


def _get_symbol_name(val: Any) -> Optional[str]:
    """Extract the symbol name (e.g. 's77') from a SymInt."""
    if _is_symint(val) and hasattr(val.node, "str"):
        return val.node.str()
    return None


def _eval_symint_range(
    val: Any, sym_ranges: Dict[str, Tuple[int, int]]
) -> Tuple[int, int, int]:
    """Evaluate a SymInt at min/opt/max bounds of its constituent symbols.

    For a base symbol like ``s77``, returns ``(lower, trace_val, upper)``
    directly from ``sym_ranges``.  For a derived expression like
    ``s77 + 3`` or ``s77 // 2``, evaluates the sympy expression at the
    lower/upper bounds of each free symbol.

    Returns:
        ``(lo, opt, hi)`` — the min, trace-time, and max values.
    """
    opt = int(val)
    sn = getattr(val, "node", None)
    if sn is None:
        logger.debug(f"[TensorRT] _eval_symint_range: no node, returning ({opt}, {opt}, {opt})")
        return opt, opt, opt

    expr = getattr(sn, "expr", None)
    if expr is None:
        logger.debug(f"[TensorRT] _eval_symint_range: no expr, returning ({opt}, {opt}, {opt})")
        return opt, opt, opt

    free_syms = expr.free_symbols
    if not free_syms:
        # Constant expression.
        return opt, opt, opt

    min_subs = {}
    max_subs = {}
    for sym in free_syms:
        sym_name = str(sym)
        if sym_name in sym_ranges:
            lo, hi = sym_ranges[sym_name]
            min_subs[sym] = lo
            # hi may be None (unbounded from Dim.AUTO); cap at opt.
            max_subs[sym] = hi if hi is not None else opt
        else:
            # Unknown symbol — pin to trace-time value.
            logger.debug(f"[TensorRT] _eval_symint_range: symbol '{sym_name}' not in sym_ranges {list(sym_ranges.keys())}")
            min_subs[sym] = opt
            max_subs[sym] = opt

    try:
        v_at_min = int(expr.subs(min_subs))
        v_at_max = int(expr.subs(max_subs))
        logger.debug(
            f"[TensorRT] _eval_symint_range: expr={expr}, "
            f"min_subs={min_subs}, max_subs={max_subs}, "
            f"result=({min(v_at_min, v_at_max)}, {opt}, {max(v_at_min, v_at_max)})"
        )
        # Expression may not be monotonic, ensure lo <= hi.
        return min(v_at_min, v_at_max), opt, max(v_at_min, v_at_max)
    except Exception as e:
        logger.debug(f"[TensorRT] _eval_symint_range: exception {e}, returning ({opt}, {opt}, {opt})")
        return opt, opt, opt


def _eval_symint_range_from_shape_env(
    val: Any,
    opt: int,
    sym_ranges: Dict[str, Tuple[int, Optional[int]]],
) -> Tuple[int, Optional[int]]:
    """Recover dynamic dim range from the SymInt's shape_env.

    After partitioning, SymInt expressions may be concretized (expr has no
    free symbols), but the shape_env still holds ``var_to_val`` which maps
    sympy variables to their trace-time values.  We use this to match
    back to ``sym_ranges`` (from ``range_constraints``).

    For derived expressions where ``opt`` doesn't directly match any
    variable hint (e.g., ``s18 // 2`` → opt=2500 vs hint=5000), we
    scale the range proportionally.

    Returns:
        ``(lo, hi)`` if a matching variable is found, otherwise ``(opt, opt)``.
    """
    sn = getattr(val, "node", None)
    if sn is None:
        return opt, opt

    shape_env = getattr(sn, "shape_env", None)
    if shape_env is None:
        return opt, opt

    var_to_val = getattr(shape_env, "var_to_val", {})

    # Direct match: find a variable whose hint equals opt and whose
    # name appears in sym_ranges with real bounds.
    for sym, hint in var_to_val.items():
        sym_name = str(sym)
        if int(hint) == opt and sym_name in sym_ranges:
            lo, hi = sym_ranges[sym_name]
            if lo != opt or hi != opt:
                return lo, hi

    # Try to find the original symbolic expression via shape_env.replacements.
    # After partitioning, sn.expr may be concretized to a constant, but
    # shape_env.replacements can map variables back to their original
    # expressions in terms of base symbols.
    expr = getattr(sn, "expr", None)
    replacements = getattr(shape_env, "replacements", {})

    # Derived expression match: opt is a scaling of a base variable.
    # E.g., opt=2500, hint=5000 → ratio = 0.5.
    for sym, hint in var_to_val.items():
        hint_val = int(hint)
        sym_name = str(sym)
        if hint_val > 0 and opt > 0 and sym_name in sym_ranges:
            sym_lo, sym_hi = sym_ranges[sym_name]
            if sym_lo == hint_val and sym_hi == hint_val:
                continue  # No real range, skip

            # Try exact evaluation: check if sn.expr still has free symbols
            # (not concretized), or look in shape_env.replacements for the
            # original expression.
            eval_expr = None
            if expr is not None and sym in getattr(expr, "free_symbols", set()):
                eval_expr = expr
            else:
                # Search replacements for an expression involving sym.
                for repl_sym, repl_expr in replacements.items():
                    if sym in getattr(repl_expr, "free_symbols", set()):
                        try:
                            if int(repl_expr.subs({sym: hint_val})) == opt:
                                eval_expr = repl_expr
                                break
                        except Exception:
                            continue

            if eval_expr is not None:
                try:
                    v_lo = int(eval_expr.subs({sym: sym_lo}))
                    v_hi = int(eval_expr.subs({sym: sym_hi})) if sym_hi is not None else None
                    return max(1, min(v_lo, v_hi if v_hi is not None else v_lo)), \
                           max(v_lo, v_hi) if v_hi is not None else None
                except Exception:
                    pass  # Fall through to conservative range

            # Proportional scaling is inaccurate for integer-division
            # formulas (e.g., (s18-1)//8+1 vs s18//8 differ by 1).
            # Use min=1 as a safe lower bound — TRT broadcasts correctly
            # with min=1, and the actual runtime value is always >= 1.
            hi = int(sym_hi * (opt / hint_val)) if sym_hi is not None else None
            return 1, hi

    return opt, opt


def _add_network_inputs(
    network: Any,
    input_nodes: List[torch.fx.Node],
    dtype_converter: Any,
) -> Dict[torch.fx.Node, Any]:
    """Add input tensors to TensorRT network.

    SymInt dimensions are converted to -1 for TRT dynamic shapes.
    """
    input_map: Dict[torch.fx.Node, Any] = {}

    for input_node in input_nodes:
        shape, dtype = _get_tensor_shape_and_dtype(input_node)
        if shape is None:
            # Symbolic sizes (e.g. sym_size from dynamic shapes) are not tensors.
            # Treat them as scalar int32 inputs with shape (1,).
            shape = (1,)
            dtype = torch.int32

        # TensorRT does not support 0-dim (scalar) tensors. Reshape to (1,).
        if len(shape) == 0:
            shape = (1,)

        # Convert SymInt dims to -1 for TRT dynamic shapes.
        trt_shape = tuple(-1 if _is_symint(s) else int(s) for s in shape)

        trt_dtype = dtype_converter(dtype if dtype else torch.float32)
        trt_input = network.add_input(
            name=input_node.name,
            dtype=trt_dtype,
            shape=trt_shape,
        )
        if trt_input is None:
            raise RuntimeError(f"Failed to add input to network: {input_node.name}")

        input_map[input_node] = trt_input

    return input_map


def _add_optimization_profile(
    builder: Any,
    config: Any,
    network: Any,
    input_nodes: List[torch.fx.Node],
    exported_program: ExportedProgram,
    trt: Any,
    symint_exprs: Optional[Dict[str, list]] = None,
) -> None:
    """Create an optimization profile for dynamic inputs.

    Must be called after the network is fully built (all layers added,
    outputs marked) so TRT can classify inputs as regular vs shape tensors.

    For regular tensor inputs with dynamic dims: ``set_shape(min, opt, max)``.
    For shape tensor inputs (values affect shapes): ``set_shape_input(min, opt, max)``.
    """
    # Check if any network input has dynamic dims or is a shape tensor.
    has_dynamic = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        dims = inp.shape
        for d in range(dims.__len__()):
            if dims[d] == -1:
                has_dynamic = True
                break
        if hasattr(inp, "is_shape_tensor") and inp.is_shape_tensor:
            has_dynamic = True
        if has_dynamic:
            break

    if not has_dynamic:
        return

    # Build symbol → (lower, upper) lookup from range_constraints and
    # shape_env.  After partitioning, SymInt expressions may be concretized
    # (expr=5000, free_symbols={}), but the shape_env attached to SymInt
    # nodes still carries the original variable ranges.  We extract ranges
    # from both sources and merge them.
    sym_ranges: Dict[str, Tuple[int, Optional[int]]] = {}

    # Source 1: range_constraints on the exported program.
    # Dim.AUTO may produce unbounded ranges (int_oo) — handle gracefully.
    if hasattr(exported_program, "range_constraints"):
        for sym, vr in exported_program.range_constraints.items():
            try:
                lo = int(vr.lower)
            except Exception:
                lo = 2  # sympy Infinity fallback
            try:
                hi = int(vr.upper)
            except Exception:
                hi = None  # mark as unbounded; resolved per-dim below
            sym_ranges[str(sym)] = (lo, hi)

    # Source 2: shape_env from SymInt metadata on placeholder nodes.
    # This survives partitioning even when SymInt expressions are concretized.
    for node in input_nodes:
        if "val" not in node.meta:
            continue
        val = node.meta["val"]
        shape = getattr(val, "shape", None)
        if shape is None:
            continue
        for s in shape:
            if not _is_symint(s):
                continue
            sn = getattr(s, "node", None)
            if sn is None:
                continue
            shape_env = getattr(sn, "shape_env", None)
            if shape_env is None:
                continue
            var_to_range = getattr(shape_env, "var_to_range", {})
            for sym, rng in var_to_range.items():
                sym_name = str(sym)
                if sym_name in sym_ranges:
                    continue  # Already have it from range_constraints
                try:
                    lo = int(rng.lower)
                except Exception:
                    lo = 2
                try:
                    hi = int(rng.upper)
                except Exception:
                    hi = None
                sym_ranges[sym_name] = (lo, hi)
            break  # Only need one shape_env — they're all the same

    logger.debug(f"[TensorRT] sym_ranges = {sym_ranges}")

    # Build node name → FX Node lookup for metadata access.
    node_lookup: Dict[str, torch.fx.Node] = {}
    for node in input_nodes:
        node_lookup[node.name] = node

    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        dims = inp.shape
        ndims = dims.__len__()
        is_shape = hasattr(inp, "is_shape_tensor") and inp.is_shape_tensor

        if is_shape:
            # Shape tensor input: set value ranges.
            # These are typically scalar int32 values (ndims=1, shape=(1,)).
            fx_node = node_lookup.get(name)
            if fx_node is not None:
                val = fx_node.meta.get("val")
                if _is_symint(val):
                    lo, opt, hi = _eval_symint_range(val, sym_ranges)
                    profile.set_shape_input(name, [lo], [opt], [hi])
                    continue
                elif isinstance(val, (int, float)):
                    v = int(val)
                    profile.set_shape_input(name, [v], [v], [v])
                    continue

            # Fallback for shape inputs: use trace-time value if available,
            # otherwise use a conservative range.
            opt_val = 1
            if fx_node is not None:
                val = fx_node.meta.get("val")
                if val is not None:
                    try:
                        opt_val = int(val)
                    except (TypeError, ValueError):
                        pass
            min_vals = [max(0, opt_val)] * ndims
            opt_vals = [max(1, opt_val)] * ndims
            max_vals = [max(1, opt_val)] * ndims
            profile.set_shape_input(name, min_vals, opt_vals, max_vals)
        else:
            # Regular tensor input: set shape ranges.
            fx_node = node_lookup.get(name)

            # Check if this input has any dynamic dims.
            has_neg = any(dims[d] == -1 for d in range(ndims))
            if not has_neg:
                # Fully static — no profile entry needed for this input.
                continue

            min_shape = []
            opt_shape = []
            max_shape = []

            raw_shape = None
            if fx_node is not None:
                raw_shape, _ = _get_tensor_shape_and_dtype(fx_node)

            for d in range(ndims):
                if dims[d] != -1:
                    # Static dim.
                    min_shape.append(dims[d])
                    opt_shape.append(dims[d])
                    max_shape.append(dims[d])
                else:
                    # Dynamic dim: get range from SymInt metadata.
                    lo, hi, opt_val = 1, 2**20, 1
                    if raw_shape is not None and d < len(raw_shape):
                        s = raw_shape[d]
                        if _is_symint(s):
                            opt_val = int(s)
                            # First try _eval_symint_range which handles
                            # both base symbols and derived expressions.
                            lo, _, hi = _eval_symint_range(s, sym_ranges)
                            # If eval returned (opt, opt, opt) because the
                            # expr was concretized, try direct symbol name
                            # lookup as fallback.
                            if lo == opt_val and hi == opt_val:
                                sym_name = _get_symbol_name(s)
                                if sym_name and sym_name in sym_ranges:
                                    lo, hi = sym_ranges[sym_name]
                            # Try stashed SymInt expressions from the
                            # partitioner (saved before concretization).
                            if lo == opt_val and hi == opt_val:
                                stashed = (fx_node.meta.get("trt_symint_exprs", {})
                                           if fx_node is not None else {})
                                stashed_expr = stashed.get(d)
                                if stashed_expr is not None:
                                    try:
                                        free = stashed_expr.free_symbols
                                        subs_min = {}
                                        subs_max = {}
                                        for sym in free:
                                            sname = str(sym)
                                            if sname in sym_ranges:
                                                slo, shi = sym_ranges[sname]
                                                subs_min[sym] = slo
                                                subs_max[sym] = shi if shi is not None else opt_val
                                        if subs_min:
                                            v_lo = int(stashed_expr.subs(subs_min))
                                            v_hi = int(stashed_expr.subs(subs_max))
                                            lo = max(1, min(v_lo, v_hi))
                                            hi = max(v_lo, v_hi)
                                    except Exception:
                                        pass  # Fall through to shape_env fallback
                            # Last resort: use shape_env.var_to_val to match
                            # back to sym_ranges from range_constraints.
                            if lo == opt_val and hi == opt_val:
                                lo, hi = _eval_symint_range_from_shape_env(
                                    s, opt_val, sym_ranges
                                )
                        else:
                            opt_val = int(s)
                            lo = opt_val
                            hi = opt_val
                    # If upper bound is unbounded (Dim.AUTO with no max),
                    # cap at the trace-time value to avoid OOM during
                    # TRT tactic search.
                    if hi is None:
                        hi = opt_val
                        logger.warning(
                            f"[TensorRT] Unbounded dynamic dim {d} for "
                            f"input '{name}': capping max at trace-time "
                            f"value {opt_val}"
                        )
                    min_shape.append(lo)
                    opt_shape.append(opt_val)
                    max_shape.append(hi)

            print(
                f"[TensorRT] Profile for '{name}': "
                f"min={tuple(min_shape)}, opt={tuple(opt_shape)}, "
                f"max={tuple(max_shape)}"
            )
            profile.set_shape(
                name,
                tuple(min_shape),
                tuple(opt_shape),
                tuple(max_shape),
            )

    config.add_optimization_profile(profile)


def _process_graph_nodes(
    graph_module: torch.fx.GraphModule,
    exported_program: ExportedProgram,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    get_trt_tensor_fn: Any,
    get_op_name_fn: Any,
    ctx: Any = None,
) -> None:
    """Process graph nodes and convert to TensorRT layers.

    Args:
        graph_module: The FX graph module to process.
        exported_program: The ExportedProgram for weight extraction.
        network: TensorRT network definition.
        input_map: Mapping from FX nodes to TensorRT tensors.
        get_trt_tensor_fn: Function to create TensorRT constant tensors.
        get_op_name_fn: Function to extract operation name from nodes.
        ctx: Optional ConversionContext for unique layer naming.
    """
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            op_name = get_op_name_fn(node)

            converter = lookup_converter(op_name)
            if converter is None:
                raise RuntimeError(f"No converter registered for operation: {op_name}")

            # Check if converter needs edge_program for weight extraction
            try:
                if needs_edge_program(op_name):
                    output_tensor = converter(node, network, input_map, exported_program, ctx)
                else:
                    output_tensor = converter(node, network, input_map, ctx)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert node '{node.name}' (op={op_name}): {e}"
                ) from e

            input_map[node] = output_tensor

            # Log convolution layer details for debugging Cask errors
            if "convolution" in op_name:
                in_shape = tuple(input_map[node.args[0]].shape) if isinstance(node.args[0], torch.fx.Node) and node.args[0] in input_map else "?"
                out_shape = tuple(output_tensor.shape) if hasattr(output_tensor, "shape") else "?"
                groups = node.args[8] if len(node.args) > 8 else "?"
                transposed = node.args[6] if len(node.args) > 6 else "?"
                w_node = node.args[1]
                w_shape = "?"
                if isinstance(w_node, torch.fx.Node) and "val" in w_node.meta:
                    wv = w_node.meta["val"]
                    if hasattr(wv, "shape"):
                        w_shape = tuple(wv.shape)
                logger.warning(
                    f"[TRT-CONV] {node.name}: in={in_shape}, out={out_shape}, "
                    f"weight={w_shape}, groups={groups}, transposed={transposed}"
                )

        elif node.op == "get_attr":
            attr_name = node.target
            param = _get_attr_value(graph_module, attr_name)
            if param is not None:
                input_map[node] = get_trt_tensor_fn(
                    network, param, f"param_{node.name}"
                )


def _mark_network_outputs(
    network: Any,
    output_nodes: List[torch.fx.Node],
    input_map: Dict[torch.fx.Node, Any],
) -> None:
    """Mark network outputs in TensorRT network."""
    for output_node in output_nodes:
        if output_node not in input_map:
            raise RuntimeError(
                f"Output node not found in input_map: {output_node.name}"
            )

        output_tensor = input_map[output_node]
        if hasattr(output_tensor, "name"):
            output_tensor.name = f"output_{output_node.name}"
        network.mark_output(output_tensor)


def _trt_dtype_to_string(dtype: Any) -> str:
    """Convert TensorRT DataType to string representation."""
    dtype_name = str(dtype)
    # dtype looks like "DataType.FLOAT" or "DataType.HALF"
    if "." in dtype_name:
        dtype_name = dtype_name.split(".")[-1]

    dtype_map = {
        "FLOAT": "float32",
        "HALF": "float16",
        "INT8": "int8",
        "INT32": "int32",
        "INT64": "int64",
        "BOOL": "bool",
        "UINT8": "uint8",
        "FP8": "float8",
        "BF16": "bfloat16",
    }
    return dtype_map.get(dtype_name, "float32")


def _collect_io_bindings(network: Any) -> List[TensorRTIOBinding]:
    """Collect I/O binding information from TensorRT network.

    Args:
        network: TensorRT network definition.

    Returns:
        List of TensorRTIOBinding with input/output tensor metadata.
    """
    # Import here to avoid circular imports at module level
    from executorch.backends.nvidia.tensorrt.converter_utils import get_safe_shape

    bindings = []

    # Collect inputs
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        is_shape = hasattr(tensor, "is_shape_tensor") and tensor.is_shape_tensor
        logger.warning(
            f"[TRT IO] input[{i}]: name={tensor.name}, dtype={tensor.dtype}, "
            f"shape={get_safe_shape(tensor)}, is_shape_tensor={is_shape}"
        )
        bindings.append(
            TensorRTIOBinding(
                name=tensor.name,
                dtype=_trt_dtype_to_string(tensor.dtype),
                shape=get_safe_shape(tensor),
                is_input=True,
                is_shape_tensor=is_shape,
            )
        )

    # Collect outputs
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        is_shape = hasattr(tensor, "is_shape_tensor") and tensor.is_shape_tensor
        bindings.append(
            TensorRTIOBinding(
                name=tensor.name,
                dtype=_trt_dtype_to_string(tensor.dtype),
                shape=get_safe_shape(tensor),
                is_input=False,
                is_shape_tensor=is_shape,
            )
        )

    return bindings


def _create_builder_config(builder: Any, spec: TensorRTCompileSpec, trt: Any) -> Any:
    """Create and configure TensorRT builder config."""
    config = builder.create_builder_config()
    if config is None:
        raise RuntimeError("Failed to create TensorRT builder config")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, spec.workspace_size)

    # Disable TF32 for strict FP32 precision on Ampere+ GPUs.
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    # Report build progress if TRT supports IProgressMonitor.
    if hasattr(trt, "IProgressMonitor"):

        class _ProgressMonitor(trt.IProgressMonitor):
            def __init__(self):
                self._seen = set()
    # Report build progress if TRT supports IProgressMonitor.
    if hasattr(trt, "IProgressMonitor"):

        class _ProgressMonitor(trt.IProgressMonitor):
            def __init__(self):
                trt.IProgressMonitor.__init__(self)
                self._seen = set()

            def phase_start(self, phase_name, parent_phase, num_steps):
                key = (phase_name, parent_phase)
                if key not in self._seen:
                    self._seen.add(key)
                    indent = "    " if parent_phase else "  "
                    print(f"{indent}TRT: {phase_name}", flush=True)

            def step_complete(self, phase_name, step):
                return True

            def phase_finish(self, phase_name):
                pass

        config.progress_monitor = _ProgressMonitor()

    # TensorRT 10.6+ enables WEIGHT_STREAMING by default, which generates
    # weight-separated plan files that require IStreamReader for deserialization.
    # We disable this flag to generate standard plan files that can be
    # deserialized with the simpler deserializeCudaEngine(data, size) API.
    if hasattr(trt.BuilderFlag, "WEIGHT_STREAMING"):
        config.clear_flag(trt.BuilderFlag.WEIGHT_STREAMING)

    # Enable cuDNN as additional tactic source.
    if hasattr(trt, "TacticSource"):
        config.set_tactic_sources(
            1 << int(trt.TacticSource.CUBLAS)
            | 1 << int(trt.TacticSource.CUBLAS_LT)
            | 1 << int(trt.TacticSource.CUDNN)
            | 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
            | 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
        )


    if spec.precision == TensorRTPrecision.FP16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.warning("FP16 not supported on this platform, using FP32")

    if spec.precision == TensorRTPrecision.INT8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            logger.warning("INT8 not supported on this platform, using FP32")

    if spec.strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    if spec.dla_core >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = spec.dla_core
        if spec.allow_gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    return config
