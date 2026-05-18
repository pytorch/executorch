# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import operator
from collections.abc import Mapping, Sequence
from typing import Any, cast, Optional, Union

import torch
from torch._inductor.decomposition import remove_decompositions
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer

logger: logging.Logger = logging.getLogger(__name__)
QuantArgs = tuple[float, int, int, int, torch.dtype]


@torch.no_grad()
def trace(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    is_qat: bool = False,
    strict: bool = False,
    ops_to_keep: Optional[list[torch._ops.OpOverload]] = None,
) -> torch.export.ExportedProgram:
    if is_qat:
        model.train()
    else:
        model.eval()

    decomp_table = torch.export.default_decompositions()
    ops_to_keep = [*(ops_to_keep or []), torch.ops.aten._safe_softmax.default]
    # pyre-fixme[6]: For 1st argument expected `Dict[typing.Callable[..., typing.Any
    remove_decompositions(decomp_table, ops_to_keep)
    program = torch.export.export(model, inputs, strict=strict).run_decompositions(
        decomp_table
    )

    return program


def prepare(
    traced_program: torch.export.ExportedProgram,
    quantizer: Quantizer,
    is_qat: bool = False,
) -> torch.fx.GraphModule:
    traced_model = traced_program.module()
    assert isinstance(traced_model, torch.fx.GraphModule)

    if is_qat:
        prepared_model = prepare_qat_pt2e(traced_model, quantizer)
    else:
        prepared_model = prepare_pt2e(traced_model, quantizer)

    return prepared_model


def extract_input_shapes_from_graph(
    module: GraphModule,
) -> dict[int, tuple[int, ...]]:
    """
    Extract input shapes from the FX graph placeholder nodes.

    Returns a dict mapping input index to expected shape tuple.
    """
    input_shapes: dict[int, tuple[int, ...]] = {}
    idx = 0
    for node in module.graph.nodes:
        if node.op == "placeholder":
            # Get the tensor_meta from the node if available
            if "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    input_shapes[idx] = tuple(val.shape)
                elif hasattr(val, "shape"):
                    input_shapes[idx] = tuple(val.shape)
            idx += 1
    return input_shapes


def extract_quant_params_through_permute(
    module: torch.fx.GraphModule,
) -> dict[int, tuple[float, int, int, int, torch.dtype]]:
    """
    Extract quantization parameters for inputs that go through a permute.

    For models with nhwc input -> conv, the graph looks like:
        x (placeholder) -> permute -> quantize -> dequantize -> conv ...
    """
    quant_args: dict[int, tuple[float, int, int, int, torch.dtype]] = {}

    placeholder_idx = 0
    for node in module.graph.nodes:
        if node.op != "placeholder":
            continue
        for user in node.users:
            if user.target in (
                torch.ops.aten.permute.default,
                torch.ops.aten.permute_copy.default,
            ):
                for permute_user in user.users:
                    target_str = str(permute_user.target)
                    if "quantize_per_tensor" in target_str:
                        args = permute_user.args[1:]
                        if len(args) >= 5:
                            quant_args[placeholder_idx] = (
                                float(args[0]),  # scale
                                int(args[1]),  # zero_point
                                int(args[2]),  # qmin
                                int(args[3]),  # qmax
                                args[4],  # dtype
                            )
                        break
                break

        placeholder_idx += 1

    return quant_args


def extract_output_dequant_params(
    module: torch.fx.GraphModule,
) -> QuantArgs:
    """
    Extract dequantization parameters from the output of a quantized model.

    The graph is expected to end with:
        ... → dequantize_per_tensor(scale, zp, qmin, qmax, dtype) → output
    """
    for node in module.graph.nodes:
        if node.op != "output":
            continue
        output_args = node.args[0]
        if isinstance(output_args, (tuple, list)):
            target_output = output_args[0]
        else:
            target_output = output_args
        if not isinstance(target_output, torch.fx.Node):
            raise ValueError("Output node is not an FX node")
        if "dequantize_per_tensor" in str(target_output.target):
            args = target_output.args[1:]
            if len(args) >= 5:
                dtype = args[4]
                assert isinstance(dtype, torch.dtype)
                return (
                    float(args[0]),  # scale
                    int(args[1]),  # zero_point
                    int(args[2]),  # qmin
                    int(args[3]),  # qmax
                    dtype,
                )
    raise ValueError("Could not find dequantize_per_tensor at the output of the graph")


def extract_output_dequant_params_through_permute(
    module: torch.fx.GraphModule,
) -> QuantArgs:
    """
    Extract dequantization parameters from the output through a permute.

    For models with nhwc output, the graph ends with:
        ... → dequantize_per_tensor → permute(0, 2, 3, 1) → output
    """
    for node in module.graph.nodes:
        if node.op != "output":
            continue
        output_args = node.args[0]
        if isinstance(output_args, (tuple, list)):
            target_output = output_args[0]
        else:
            target_output = output_args
        if not isinstance(target_output, torch.fx.Node):
            raise ValueError("Output node is not an FX node")
        if target_output.target in (
            torch.ops.aten.permute.default,
            torch.ops.aten.permute_copy.default,
        ):
            permute_input = target_output.args[0]
            if isinstance(
                permute_input, torch.fx.Node
            ) and "dequantize_per_tensor" in str(permute_input.target):
                args = permute_input.args[1:]
                if len(args) >= 5:
                    dtype = args[4]
                    assert isinstance(dtype, torch.dtype)
                    return (
                        float(args[0]),  # scale
                        int(args[1]),  # zero_point
                        int(args[2]),  # qmin
                        int(args[3]),  # qmax
                        dtype,
                    )
    raise ValueError(
        "Could not find dequantize_per_tensor → permute at the output of the graph"
    )


def extract_input_quant_params_from_graph(
    module: GraphModule,
    input_names: list[str],
) -> dict[int, QuantArgs]:
    """
    Extract quantization parameters from the FX graph for model inputs.
    """
    quant_args: dict[int, QuantArgs] = {}
    found_names: set[str] = set()

    if not input_names:
        return quant_args

    for idx, name in enumerate(input_names):
        for node in module.graph.nodes:
            if node.op != "call_function":
                continue

            if (
                node.args
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].name == name
                and not node.name.startswith("_assert_tensor_metadata")
                and "quantize_per_tensor" in str(node.target)
            ):
                args = node.args[1:]
                if len(args) >= 5:
                    quant_args[idx] = (
                        float(args[0]),  # scale
                        int(args[1]),  # zero_point
                        int(args[2]),  # qmin
                        int(args[3]),  # qmax
                        args[4],  # dtype
                    )
                    found_names.add(name)
                break

    missing_names = set(input_names) - found_names
    if missing_names:
        raise ValueError(
            f"Could not find quantization parameters for input(s): {sorted(missing_names)}. "
            f"Make sure these input names exist in the graph and quantization parameters."
        )

    return quant_args


class QuantizedInputWrapper(torch.nn.Module):
    """
    Wrapper that allows a quantized model to accept quantized inputs.

    If no input_names or quant_args are provided, the wrapper passes inputs
    through unchanged (no dequantization).

    Args:
        module: The quantized GraphModule to wrap.
        input_names: Optional list of input placeholder names in the graph.
            If provided, extracts quant params from graph.
        quant_args: Optional dict mapping input index to (scale, zero_point, qmin, qmax, dtype).
            If provided, uses these directly instead of extracting from graph.
        expected_inputs: Optional dict mapping input index to the expected
            dequantized tensor. After dequantization, the result is compared
            against these values using atol/rtol. Raises ValueError if exceeded.
        atol: Absolute tolerance for the expected-value check (default 1e-4).
        rtol: Relative tolerance for the expected-value check (default 1e-4).

    Example:
        # Extract from graph
        wrapper = QuantizedInputWrapper(quantized_module, input_names=["x"])

        # Explicit quant args with expected-value validation
        wrapper = QuantizedInputWrapper(
            quantized_module,
            quant_args={0: (1/255, 0, 0, 255, torch.uint8)},
            expected_inputs={0: reference_float_tensor},
            atol=1e-3,
        )
    """

    def __init__(
        self,
        module: GraphModule,
        input_args: Optional[Union[list[str], dict[int, QuantArgs]]] = None,
        expected_inputs: Optional[dict[int, torch.Tensor]] = None,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.module: GraphModule = module
        self.quant_args: dict[int, QuantArgs] = {}
        self.expected_shapes: dict[int, tuple[int, ...]] = (
            extract_input_shapes_from_graph(module)
        )
        self.expected_inputs: Optional[dict[int, torch.Tensor]] = expected_inputs
        self.atol: float = atol
        self.rtol: float = rtol

        if input_args is not None:
            logger.warning(
                "Warning: Using pre-quantized inputs. This should only be done when calibration has been confirmed."
                "Incorrect quantization parameters can lead to significant accuracy degradation."
            )
        if isinstance(input_args, Sequence) and not isinstance(
            input_args, (str, bytes)
        ):
            self.quant_args = extract_input_quant_params_from_graph(
                module, list(input_args)
            )
        elif isinstance(input_args, Mapping):
            # dict[int, QuantArgs] — use directly
            # dict[int, list[str]] — extract quant params from graph, keyed by input index
            first_value = next(iter(input_args.values()), None)
            if (
                isinstance(first_value, (list, tuple, Sequence))
                and not isinstance(first_value, (str, bytes))
                and first_value
                and isinstance(first_value[0], str)
            ):
                # Values are lists of node names: extract quant params and map
                # to the caller-specified input indices.
                for input_idx, node_names in input_args.items():
                    extracted = extract_input_quant_params_from_graph(
                        module, list(cast(Sequence[str], node_names))
                    )
                    # Use the first extracted quant params for this input index.
                    if extracted:
                        self.quant_args[int(input_idx)] = next(iter(extracted.values()))
            else:
                self.quant_args = {int(k): v for k, v in input_args.items()}

    def forward(self, *args: torch.Tensor) -> Any:
        """Run inference, dequantizing configured inputs."""
        # Validate input shapes for quantized inputs
        for index in self.quant_args:
            if index >= len(args):
                continue
            actual_shape = tuple(args[index].shape)
            if index not in self.expected_shapes:
                continue
            expected_shape = self.expected_shapes[index]
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for quantized input at index {index}: "
                    f"expected {expected_shape}, got {actual_shape}"
                )

        dequantized_args = []
        for index, node in enumerate(args):
            if index in self.quant_args:
                scale, zp, qmin, qmax, dtype = self.quant_args[index]
                node = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    node, scale, zp, qmin, qmax, dtype
                )
            dequantized_args.append(node)

        # Check dequantized values against expected inputs
        expected_inputs = self.expected_inputs
        if expected_inputs is not None:
            for index, expected in expected_inputs.items():
                if index >= len(dequantized_args):
                    continue
                actual = dequantized_args[index]
                if not torch.allclose(actual, expected, atol=self.atol, rtol=self.rtol):
                    max_abs_diff = (actual - expected).abs().max().item()
                    mean_abs_diff = (actual - expected).abs().mean().item()
                    msg = (
                        f"Dequantized input at index {index} differs from expected value: "
                        f"max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g} "
                        f"(atol={self.atol}, rtol={self.rtol})"
                    )
                    raise ValueError(msg)

        return self.module(*dequantized_args)

    @staticmethod
    def sink_dequants(program: torch.export.ExportedProgram) -> None:
        """Sink dequant nodes through transparent ops in an exported program.

        If the graph branches through transparent ops (view, split, getitem, etc.)
        into paths with different quantization parameters, sink the dequants to be
        adjacent to each downstream quant node, enabling per-branch fusion.

        Must be called after export() on a QuantizedInputWrapper-wrapped model.
        """
        from torch.export.graph_signature import InputKind

        user_input_names = {
            spec.arg.name
            for spec in program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
        }
        sink_input_dequant_through_transparent_ops(
            program.graph_module, user_input_names
        )


class QuantizedOutputWrapper(torch.nn.Module):
    """
    Wrapper that quantizes a model's output so it produces uint8 tensors.

    Mirrors QuantizedInputWrapper: the wrapper adds a quantize_per_tensor after
    the model's output. When the graph is traced, the dequant (from the model) →
    quant (from the wrapper) pair with matching parameters folds away, leaving
    the output in its quantized form.

    Args:
        module: The module to wrap (may already be a QuantizedInputWrapper).
        output_quant_args: (scale, zero_point, qmin, qmax, dtype) for the output.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        output_quant_args: QuantArgs,
    ) -> None:
        super().__init__()
        self.module: torch.nn.Module = module
        self.output_quant_args: QuantArgs = output_quant_args

    def forward(self, *args: torch.Tensor) -> Any:
        result = self.module(*args)
        scale, zp, qmin, qmax, dtype = self.output_quant_args
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            result, scale, zp, qmin, qmax, dtype
        )


def _get_transparent_ops() -> set[Any]:
    """Ops that only reshape/index data without changing values.
    Safe to pass uint8 data through these."""
    return {
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.chunk.default,
        torch.ops.aten.slice_copy.Tensor,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand_copy.default,
        torch.ops.aten.unsqueeze_copy.default,
        torch.ops.aten.squeeze_copy.dim,
        torch.ops.aten.transpose_copy.int,
        torch.ops.aten.clone.default,
        operator.getitem,
    }


def _get_quantize_ops() -> set[Any]:
    ops = {torch.ops.quantized_decomposed.quantize_per_tensor.default}
    try:
        ops.add(torch.ops.cadence.quantize_per_tensor.default)
    except AttributeError:
        pass
    return ops


def _get_dequantize_ops() -> set[Any]:
    ops = {torch.ops.quantized_decomposed.dequantize_per_tensor.default}
    try:
        ops.add(torch.ops.cadence.dequantize_per_tensor.default)
    except AttributeError:
        pass
    return ops


def _walk_to_downstream_quants(
    node: torch.fx.Node,
    quantize_ops: set[Any],
    transparent_ops: set[Any],
    downstream_quants: list[torch.fx.Node],
) -> bool:
    """Walk forward through transparent ops collecting downstream quant nodes.

    Returns True if all paths end at a quant node.
    """
    all_valid = True
    for user in node.users:
        if user.op == "call_function" and user.target in quantize_ops:
            downstream_quants.append(user)
        elif user.op == "call_function" and user.target in transparent_ops:
            if not _walk_to_downstream_quants(
                user, quantize_ops, transparent_ops, downstream_quants
            ):
                all_valid = False
        else:
            all_valid = False
    return all_valid


def _get_dequant_node_for_placeholder(
    placeholder: torch.fx.Node,
    input_placeholder_names: set[str] | None,
    dequantize_ops: set[Any],
) -> torch.fx.Node | None:
    """Return the single dequant user of a uint8 placeholder, or None."""
    if placeholder.op != "placeholder":
        return None
    if (
        input_placeholder_names is not None
        and placeholder.name not in input_placeholder_names
    ):
        return None
    val = placeholder.meta.get("val")
    if val is None or not isinstance(val, torch.Tensor):
        return None
    if val.dtype != torch.uint8:
        return None
    if len(placeholder.users) != 1:
        return None
    dequant_node = next(iter(placeholder.users))
    if dequant_node.op == "call_function" and dequant_node.target in dequantize_ops:
        return dequant_node
    return None


def _sink_dequant_to_quant_nodes(
    graph: torch.fx.Graph,
    dequant_node: torch.fx.Node,
    placeholder: torch.fx.Node,
    downstream_quants: list[torch.fx.Node],
) -> None:
    """Insert per-branch dequants before each downstream quant and rewire."""
    dequant_op = dequant_node.target
    assert callable(dequant_op)

    for quant_node in downstream_quants:
        quant_input = quant_node.args[0]
        assert isinstance(quant_input, torch.fx.Node)
        quant_params = quant_node.args[1:]

        with graph.inserting_before(quant_node):
            new_dequant = graph.call_function(
                dequant_op,
                args=(quant_input, *quant_params),
            )
        new_dequant.meta = {**dequant_node.meta}
        if "val" in quant_node.meta and isinstance(
            quant_node.meta["val"], torch.Tensor
        ):
            quant_val = quant_node.meta["val"]
            new_dequant.meta["val"] = torch.empty(quant_val.shape, dtype=torch.float32)

        quant_node.replace_input_with(quant_input, new_dequant)

    dequant_node.replace_all_uses_with(placeholder)
    graph.erase_node(dequant_node)


def sink_input_dequant_through_transparent_ops(
    graph_module: GraphModule,
    input_placeholder_names: set[str] | None = None,
) -> bool:
    """
    Sinks dequantize nodes from quantized input placeholders through transparent ops
    to be adjacent to downstream quantize nodes, enabling dequant-quant fusion.
    This creates per-branch dequants with matching params.

    Args:
        graph_module: The graph module to transform.
        input_placeholder_names: Optional set of placeholder names to consider.
            If provided, only these placeholders are processed (use this to
            restrict to user inputs and avoid touching weight/buffer placeholders).
            If None, all uint8 placeholders are considered.

    Returns True if the graph was modified.
    """
    graph = graph_module.graph
    modified = False

    transparent_ops: set[Any] = _get_transparent_ops()
    quantize_ops: set[Any] = _get_quantize_ops()
    dequantize_ops: set[Any] = _get_dequantize_ops()

    for placeholder in list(graph.nodes):
        dequant_node = _get_dequant_node_for_placeholder(
            placeholder, input_placeholder_names, dequantize_ops
        )
        if dequant_node is None:
            continue

        downstream_quants: list[torch.fx.Node] = []
        all_paths_end_at_quant = _walk_to_downstream_quants(
            dequant_node, quantize_ops, transparent_ops, downstream_quants
        )

        if not downstream_quants or not all_paths_end_at_quant:
            continue

        _sink_dequant_to_quant_nodes(
            graph, dequant_node, placeholder, downstream_quants
        )

        modified = True
        logger.info(
            "Sunk dequant for input '%s' through transparent ops to %d "
            "downstream quant nodes",
            placeholder.name,
            len(downstream_quants),
        )

    if modified:
        graph.lint()
        graph_module.recompile()

    return modified
