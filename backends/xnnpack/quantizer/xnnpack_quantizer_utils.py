# mypy: allow-untyped-defs
import itertools
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from executorch.backends.xnnpack.utils.utils import (
    get_groups_from_conv,
    is_depthwise_conv,
)
from torch._subclasses import FakeTensor
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torchao.quantization.pt2e import WrapperModule
from torchao.quantization.pt2e.graph_utils import get_source_partitions
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
    OperatorConfig,
    OperatorPatternType,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e.utils import (
    _get_aten_graph_module_for_pattern,
    _is_conv_node,
    _is_conv_transpose_node,
    get_new_attr_name_with_prefix,
)

__all__ = [
    "OperatorConfig",
    "OperatorPatternType",
    "QuantizationConfig",
    "QuantizationSpec",
    "get_input_act_qspec",
    "get_output_act_qspec",
    "get_weight_qspec",
    "get_bias_qspec",
    "OP_TO_ANNOTATOR",
    "propagate_annotation",
]


AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        Optional[QuantizationConfig],
        Optional[Callable[[Node], bool]],
    ],
    Optional[list[list[Node]]],
]
OP_TO_ANNOTATOR: dict[str, AnnotatorType] = {}


def register_annotator(op: str) -> Callable[[AnnotatorType], None]:
    def decorator(annotator: AnnotatorType) -> None:
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


def change_quantization_config(
    original_qspec,
    dtype=None,
    quant_min=None,
    quant_max=None,
    qscheme=None,
    ch_axis=None,
    is_dynamic=None,
    observer_or_fake_quant_ctr=None,
):
    return QuantizationSpec(
        dtype=dtype or original_qspec.dtype,
        quant_min=quant_min or original_qspec.quant_min,
        quant_max=quant_max or original_qspec.quant_max,
        qscheme=qscheme or original_qspec.qscheme,
        ch_axis=ch_axis or original_qspec.ch_axis,
        is_dynamic=is_dynamic or original_qspec.is_dynamic,
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr
        or original_qspec.observer_or_fake_quant_ctr,
    )


def is_relu_node(node: Node) -> bool:
    """
    Check if a given node is a relu node
    """
    return node.op == "call_function" and node.target in [
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
    ]


def _is_annotated(nodes: list[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            Q_ANNOTATION_KEY in node.meta and node.meta[Q_ANNOTATION_KEY]._annotated
        )
    return annotated


def _mark_nodes_as_annotated(nodes: list[Node]):
    for node in nodes:
        if node is not None:
            if Q_ANNOTATION_KEY not in node.meta:
                node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
            node.meta[Q_ANNOTATION_KEY]._annotated = True


@register_annotator("linear")
def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[2]

        if _is_annotated([node]) is False:  # type: ignore[list-item]
            annotate_input_qspec_map(
                node,
                act_node,
                input_act_qspec,
            )
            annotate_input_qspec_map(
                node,
                weight_node,
                weight_qspec,
            )
            nodes_to_mark_annotated = [node, weight_node]
            if bias_node:
                annotate_input_qspec_map(
                    node,
                    bias_node,
                    bias_qspec,
                )
                nodes_to_mark_annotated.append(bias_node)
            annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions


@register_annotator("linear_relu")
def _annotate_linear_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if not is_relu_node(node):
            continue
        relu_node = node
        maybe_linear_node = node.args[0]
        if (
            not isinstance(maybe_linear_node, Node)
            or maybe_linear_node.op != "call_function"
            or maybe_linear_node.target != torch.ops.aten.linear.default
        ):
            continue

        linear_node = maybe_linear_node
        if len(linear_node.users) > 1:
            # if linear node has multiple users, then it can't be fused with relu
            continue

        input_qspec_map = {}
        input_act = linear_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = input_act_qspec

        weight = linear_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = weight_qspec

        # adding weight node to the partition as well
        partition = [relu_node, linear_node, weight]
        bias = linear_node.args[2] if len(linear_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = bias_qspec
            partition.append(bias)

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        linear_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


def _do_annotate_conv(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    is_conv_transpose: bool = False,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    is_conv_node = _is_conv_transpose_node if is_conv_transpose else _is_conv_node

    for n in gm.graph.nodes:
        if not is_conv_node(n):
            continue
        conv_node = n

        # This is hacky!
        # We do not want to annotate conv node independently if there is a conv + relu pattern
        # So we skip if the conv node is consumed by a single relu node
        if len(conv_node.users) == 1:
            user = list(conv_node.users.keys())[0]
            if is_relu_node(user):
                continue

        # Tracks conditions for whether or not to skip
        skip = False

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        weight_qspec = get_weight_qspec(quantization_config)
        num_groups = get_groups_from_conv(conv_node)

        # skip if transposed conv has more than 1 group
        skip = skip or (is_conv_transpose and num_groups != 1)

        if is_conv_transpose:
            # transposed convs per output channel quantization
            weight_qspec = change_quantization_config(weight_qspec, ch_axis=1)

        input_qspec_map[weight] = weight_qspec
        is_dynamic = (
            quantization_config
            and quantization_config.input_activation
            and quantization_config.input_activation.is_dynamic
        )

        # Only annotate dynamically quantized conv if it's 2D and not depthwise
        if is_dynamic:
            weight_val = weight.meta.get("val", None)
            weight_shape = getattr(weight_val, "shape", None)
            # Skip if not a 4D weight tensor (i.e. not conv2d)
            skip = skip or (weight_shape is not None and len(weight_shape) != 4)
            # Skip if depthwise (default to groups=1 since it's not an arg)
            skip = skip or (
                not is_conv_transpose and is_depthwise_conv(weight_shape, 1, False)
            )

        # adding weight node to the partition as well
        partition = [conv_node, conv_node.args[1]]

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)

        if _is_annotated(partition) or skip:
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


def _do_annotate_conv_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    is_conv_transpose: bool = False,
):
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not is_relu_node(n):
            continue
        relu_node = n
        maybe_conv_node = n.args[0]

        is_conv_node = _is_conv_transpose_node if is_conv_transpose else _is_conv_node
        if not isinstance(maybe_conv_node, Node) or not is_conv_node(maybe_conv_node):
            continue
        conv_node = maybe_conv_node

        if len(conv_node.users) > 1:
            # relu shouldn't be fuseable to conv if there are other users
            # of convolution
            continue

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        weight_qspec = get_weight_qspec(quantization_config)
        groups = get_groups_from_conv(conv_node)
        if is_conv_transpose:
            # transposed convs per output channel quantization
            weight_qspec = change_quantization_config(weight_qspec, ch_axis=1)
        input_qspec_map[weight] = weight_qspec

        # adding weight node to the partition as well
        partition = [relu_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)

        if _is_annotated(partition):
            continue

        if is_conv_transpose and groups != 1:
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )
        relu_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("conv")
def _annotate_conv(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    return _do_annotate_conv(
        gm, quantization_config, filter_fn, is_conv_transpose=False
    )


@register_annotator("conv_transpose")
def _annotate_transpose_conv(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    return _do_annotate_conv(gm, quantization_config, filter_fn, is_conv_transpose=True)


@register_annotator("conv_relu")
def _annotate_conv_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    return _do_annotate_conv_relu(
        gm, quantization_config, filter_fn, is_conv_transpose=False
    )


@register_annotator("conv_transpose_relu")
def _annotate_conv_transpose_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    return _do_annotate_conv_relu(
        gm, quantization_config, filter_fn, is_conv_transpose=True
    )


@register_annotator("conv_bn")
def _annotate_conv_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv + batchnorm parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(gm, quantization_config, filter_fn, has_relu=False)


@register_annotator("conv_bn_relu")
def _annotate_conv_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv + batchnorm + relu parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(gm, quantization_config, filter_fn, has_relu=True)


@register_annotator("conv_transpose_bn")
def _annotate_conv_transpose_bn(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv_transpose + batchnorm parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(
        gm, quantization_config, filter_fn, has_relu=False, is_conv_transpose=True
    )


@register_annotator("conv_transpose_bn_relu")
def _annotate_conv_transpose_bn_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv_transpose + batchnorm + relu parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_bn(
        gm, quantization_config, filter_fn, has_relu=True, is_conv_transpose=True
    )


def _do_annotate_conv_bn(  # noqa: C901
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]],
    has_relu: bool,
    is_conv_transpose: bool = False,
) -> list[list[Node]]:
    """
    Given a function that takes in a `conv_fn` and returns a conv-bn[-relu] pattern,
    return a list of annotated partitions.

    The output of the pattern must include a dictionary from string name to node
    for the following names: "input", "conv", "weight", "bias", and "output".
    """

    # Example inputs for conv-bn1d patterns
    _conv1d_bn_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    # Example inputs for conv-bn2d patterns
    _conv2d_bn_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    def get_pattern(conv_fn: Callable, relu_is_inplace: bool):
        def _conv_bn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_rm, bn_rv):
            conv = conv_fn(x, conv_weight, conv_bias)
            bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
            if has_relu:
                output = F.relu_(bn) if relu_is_inplace else F.relu(bn)
            else:
                output = bn
            return output, {
                "input": x,
                "conv": conv,
                "weight": conv_weight,
                "bias": conv_bias,
                "output": output,
            }

        return WrapperModule(_conv_bn)

    # Needed for matching, otherwise the matches gets filtered out due to unused
    # nodes returned by batch norm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    matches = []
    if is_conv_transpose:
        combinations = [
            (F.conv_transpose1d, _conv1d_bn_example_inputs),
            (F.conv_transpose2d, _conv2d_bn_example_inputs),
        ]
    else:
        combinations = [
            (F.conv1d, _conv1d_bn_example_inputs),  # type: ignore[list-item]
            (F.conv2d, _conv2d_bn_example_inputs),  # type: ignore[list-item]
        ]

    # Add `is_cuda` and `relu_is_inplace` dimensions
    combinations = itertools.product(  # type: ignore[assignment]
        combinations,
        [True, False] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False] if has_relu else [False],  # relu_is_inplace
    )

    # Match against all conv dimensions and cuda variants
    for (conv_fn, example_inputs), is_cuda, relu_is_inplace in combinations:  # type: ignore[misc]
        pattern = get_pattern(conv_fn, relu_is_inplace)  # type: ignore[has-type]
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda)  # type: ignore[has-type]
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        matches.extend(matcher.match(gm.graph))

    # Annotate nodes returned in the matches
    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        conv_node = name_node_map["conv"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # Validate conv args
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # Annotate conv inputs and pattern output
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        if bias_node is not None:
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        conv_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        output_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("gru_io_only")
def _annotate_gru_io_only(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    gru_partitions = list(itertools.chain.from_iterable(gru_partitions.values()))
    annotated_partitions = []
    for gru_partition in gru_partitions:
        annotated_partitions.append(gru_partition.nodes)
        output_nodes = gru_partition.output_nodes
        input_nodes = gru_partition.input_nodes
        # skip annotation if it is already annotated
        if _is_annotated(input_nodes + output_nodes):
            continue
        # inside each GRU partition, we should be able to annotate each linear
        # subgraph
        input_act = input_nodes[0]
        input_act_user = next(iter(input_act.users.keys()))
        assert isinstance(input_act, Node)
        assert isinstance(input_act_user, Node)
        input_act_user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={
                input_act: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )

        hidden_state = input_nodes[1]
        hidden_state_user = next(iter(hidden_state.users.keys()))
        assert isinstance(hidden_state, Node)
        assert isinstance(hidden_state_user, Node)
        hidden_state_user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={
                hidden_state: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )

        assert len(output_nodes) == 2, "expecting GRU to have two outputs"
        for output in output_nodes:
            output.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        nodes_to_mark_annotated = list(gru_partition.nodes)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
    return annotated_partitions


@register_annotator("adaptive_avg_pool2d")
def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """Always annotate adaptive_avg_pool2d op"""
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], filter_fn
    )
    partitions = list(itertools.chain.from_iterable(module_partitions.values()))
    annotated_partitions = []
    for partition in partitions:
        pool_node = partition.output_nodes[0]
        if (
            pool_node.op != "call_function"
            or pool_node.target != torch.ops.aten.adaptive_avg_pool2d.default
        ):
            raise ValueError(f"{pool_node} is not an aten adaptive_avg_pool2d operator")

        if _is_annotated([pool_node]):
            continue

        annotated_partitions.append(partition.nodes)
        input_act = pool_node.args[0]
        assert isinstance(input_act, Node)

        # only annotate input output sharing operator
        # when the output of the input node is annotated
        if (
            Q_ANNOTATION_KEY not in input_act.meta
            or not input_act.meta[Q_ANNOTATION_KEY]._annotated
            or input_act.meta[Q_ANNOTATION_KEY].output_qspec is None
        ):
            input_act_qspec = get_input_act_qspec(quantization_config)
        else:
            input_act_qspec = SharedQuantizationSpec(input_act)

        # output sharing with input
        output_act_qspec = SharedQuantizationSpec((input_act, pool_node))
        pool_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={
                input_act: input_act_qspec,
            },
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


def _is_input_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """Check if input is a large scalar value. So that we can skip quantization for the node
    since histc op (in HistogramObserver) only works for values up to certain upper bound
    """
    if node.op == "get_attr":
        qualified_name = str(node.target)
        module_path, _, name = qualified_name.rpartition(".")
        submod = gm.get_submodule(module_path)
        tensor = getattr(submod, name)
        # torch.histc works until this upper bound
        HISTC_UPPER_BOUND = 3.4028235e15
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False


def _is_input_non_float_tensor(node: Node):
    """Check if the input is not a float tensor, so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.float32


@register_annotator("add_relu")
def _annotate_add_relu(  # noqa: C901
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if not is_relu_node(node):
            continue
        relu_node = node
        maybe_add = node.args[0]
        if (
            not isinstance(maybe_add, Node)
            or maybe_add.op != "call_function"
            or maybe_add.target
            not in [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            ]
        ):
            continue

        add_node = maybe_add

        if len(add_node.users) > 1:
            # add can't be fused with ReLU if the result of add is being used
            # else where in the graph
            continue

        partition = [relu_node, add_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            partition.append(input_act0)
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            partition.append(input_act1)
            input_qspec_map[input_act1] = input_act_qspec

        add_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("add")
def _annotate_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        ]:
            continue
        add_node = node
        partition = [add_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            input_qspec_map[input_act0] = input_act_qspec
            partition.append(input_act0)

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            input_qspec_map[input_act1] = input_act_qspec
            partition.append(input_act1)

        add_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("mul_relu")
def _annotate_mul_relu(  # noqa: C901
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if not is_relu_node(node):
            continue
        relu_node = node
        maybe_mul = node.args[0]
        if (
            not isinstance(maybe_mul, Node)
            or maybe_mul.op != "call_function"
            or maybe_mul.target
            not in [
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul_.Tensor,
            ]
        ):
            continue

        mul_node = maybe_mul
        if len(mul_node.users) > 1:
            # mul can't be fused with ReLU if the result of mul is being used
            # else where in the graph
            continue

        partition = [relu_node, mul_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            partition.append(input_act0)
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            partition.append(input_act1)
            input_qspec_map[input_act1] = input_act_qspec

        mul_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        relu_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("mul")
def _annotate_mul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul_.Tensor,
        ]:
            continue

        mul_node = node
        partition = [mul_node]
        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            input_qspec_map[input_act0] = input_act_qspec
            partition.append(input_act0)

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            input_qspec_map[input_act1] = input_act_qspec
            partition.append(input_act0)

        mul_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        annotated_partitions.append(partition)
    return annotated_partitions


# TODO: remove Optional in return type, fix annotated_partitions logic
@register_annotator("cat")
def _annotate_cat(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for cat_node in gm.graph.nodes:
        if cat_node.target != torch.ops.aten.cat.default:
            continue

        if _is_annotated([cat_node]):
            continue

        annotated_partitions.append(cat_node.all_input_nodes)

        input_act_qspec = get_input_act_qspec(quantization_config)
        inputs = cat_node.args[0]

        input_qspec_map = {}
        input_act0 = inputs[0]  # type: ignore[index]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        shared_with_input0_qspec = SharedQuantizationSpec((input_act0, cat_node))  # type: ignore[arg-type]
        for input_act in inputs[1:]:  # type: ignore[index, union-attr]
            if input_act not in input_qspec_map:
                input_qspec_map[input_act] = shared_with_input0_qspec  # type: ignore[index]

        output_act_qspec = shared_with_input0_qspec

        cat_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


def _is_share_obs_or_fq_op(op: Callable) -> bool:
    return op in [
        torch.ops.aten.relu.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dim,
        # TODO: remove?
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.slice_copy.Tensor,
        torch.ops.aten.flatten.using_ints,
    ]


def propagate_annotation(model: torch.fx.GraphModule) -> None:
    for n in model.graph.nodes:
        if n.op != "call_function" or not _is_share_obs_or_fq_op(n.target):
            continue

        prev_node = n.args[0]
        if not isinstance(prev_node, Node):
            continue

        quantization_annotation = prev_node.meta.get(Q_ANNOTATION_KEY, None)
        if not quantization_annotation:
            continue

        output_qspec = quantization_annotation.output_qspec
        if not output_qspec:
            continue

        # make sure current node is not annotated
        if Q_ANNOTATION_KEY in n.meta and n.meta[Q_ANNOTATION_KEY]._annotated:
            continue

        shared_qspec = SharedQuantizationSpec(prev_node)
        # propagate the previous output_qspec to the current node
        n.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={
                prev_node: shared_qspec,
            },
            output_qspec=shared_qspec,
            _annotated=True,
        )


# TODO: make the list of ops customizable
def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            float_tensor = torch.tensor(float(args[i]))
            model.register_buffer(tensor_constant_name, float_tensor)
            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node(
                    "get_attr", tensor_constant_name, (), {}
                )
                get_attr_node.meta["val"] = fake_mode.from_tensor(
                    float_tensor, static_shapes=True
                )
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()
    return model
