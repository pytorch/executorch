import copy
import logging
import operator
import warnings
from typing import Any, Optional

import torch
import torch.fx
import torch.fx as fx
import torch_tensorrt.fx.tracer.acc_tracer.acc_utils as acc_utils
from torch.fx.experimental.const_fold import split_const_subgraphs

from ..observer import observable
from ..tracer.acc_tracer import acc_ops
from ..tracer.acc_tracer.acc_utils import get_attr
from .pass_utils import log_before_after, validate_inference

_LOGGER = logging.getLogger(__name__)

# Create an alias for module input type to avoid littering pyre-ignore for Any
# throughout the file.
Input = Any


def replace_mutable_op(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    if not isinstance(module, torch.fx.GraphModule):
        return module

    # Before any lowering pass, replace mutable ops like torch.fill_
    # Because fx cannot deal with inplace ops
    for n in module.graph.nodes:
        # TODO: add more mutable ops
        if (n.op == "call_method" and n.target == "fill_") or (
            n.op == "call_function" and n.target == torch.fill_
        ):
            # Replace mutable op only if the modified variable
            # is used by the rest of the graph
            # only through this op
            if set(n.args[0].users.keys()) == {n}:
                with module.graph.inserting_after(n):
                    # TODO: move this outside?
                    def fill_with_mul_zero_and_add(*args):
                        return args[0].mul(0.0).add(args[1])

                    new_node = module.graph.create_node(
                        "call_function", fill_with_mul_zero_and_add, args=n.args
                    )
                    n.replace_all_uses_with(new_node)
                    module.graph.erase_node(n)
    module.recompile()
    return module


def run_const_fold(traced_mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    def skip_folding_ops(node: torch.fx.Node):
        # dtype op
        if node.target == acc_ops.dtype:
            return True
        # Now we do constant folding on traced module. We want to skip pattern like
        # weights -> quant -> dequant -> op during constant folding when the model is
        # a quantized int8 model.
        # quant_dequant
        if node.target != acc_ops.quantize_per_tensor:
            return False
        # If quantize_per_node -> dequantize, then skip folding.
        for user in node.users:
            if user.target == acc_ops.dequantize:
                return True
        return False

    const_split_mod = split_const_subgraphs(traced_mod, skip_folding_ops)
    const_split_mod.run_folding()
    return const_split_mod


def replace_op_with_indices(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.aten.max_pool3d_with_indices.default,
            torch.ops.aten.native_batch_norm.default,
        ):
            if len(n.users) != 1:
                raise RuntimeError(
                    f"{n.target} has users={len(n.users)}. We can only handle it with 1 user"
                )
            if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                new_op = torch.ops.aten.max_pool2d
                new_args = n.args
            elif n.target == torch.ops.aten.max_pool3d_with_indices.default:
                new_op = torch.ops.aten.max_pool3d
                new_args = n.args
            elif n.target == torch.ops.aten.native_batch_norm.default:
                new_op = torch.ops.aten.batch_norm
                new_args = list(n.args)
                new_args.append(False)
                new_args = tuple(new_args)

            getitem_node = next(iter(n.users))
            with module.graph.inserting_after(getitem_node):
                new_node = module.graph.create_node(
                    "call_function",
                    new_op,
                    args=new_args,
                    kwargs=n.kwargs,
                )
                getitem_node.replace_all_uses_with(new_node)
                module.graph.erase_node(getitem_node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return module


@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def fuse_sparse_matmul_add(gm: torch.fx.GraphModule, input: Input):
    """
    Replace acc_ops.matmul + acc_ops.add with acc_ops.linear
    TRT8.2 can take advantage of structured sparsity (2:4), but the graph needs contain a single FC layer.
    Later versions of TRT should work with matmul.

    Example before:
    def forward(self, x):
        a = self.a
        b = self.b
        addmm_mm = torch_tensorrt.fx.tracer.acc_tracer.acc_ops.matmul(input = a, other = b);  a = b = None
        addmm_add = torch_tensorrt.fx.tracer.acc_tracer.acc_ops.add(input = addmm_mm, other = x);  addmm_mm = x = None
        return addmm_add

    After:
    def forward(self, x):
        a = self.a
        b = self.b
        linear_1 = torch_tensorrt.fx.tracer.acc_tracer.acc_ops.linear(input = a, weight = b, bias = x);  a = b = x = None
        return linear_1
    """
    counter = 0
    for node in gm.graph.nodes:
        if node.target != acc_ops.add:
            continue
        add_node = node
        bias = add_node.kwargs["other"]

        if bias.op != "get_attr":
            continue
        # test that bias tensor is one-dimensional, should correspond to shape (out_features)
        if get_attr(bias).dim() > 1:
            continue

        node = add_node.kwargs["input"]
        if node.target != acc_ops.matmul:
            continue
        matmul_node = node
        a = matmul_node.kwargs["input"]

        node = matmul_node.kwargs["other"]
        if node.op != "get_attr":
            continue

        get_attr_node = node
        weight = get_attr(get_attr_node)
        # TODO: verify that weight comply with TRT structured sparsity requirements:
        # For each output channel and for each spatial pixel in the kernel weights,
        # every 4 input channels must have at least 2 zeros.

        # test that weight tensor is two-dimensional, should correspond to shape (out_features, in_features)
        if weight.dim() != 2:
            continue

        weight_t = weight.transpose(0, 1)
        weight_t_name = "weight_t_tensor_" + str(counter)
        gm.register_buffer(weight_t_name, weight_t)
        counter += 1

        with gm.graph.inserting_before(add_node):
            weight_t_attr = gm.graph.get_attr(weight_t_name)
            fused_node = gm.graph.call_function(
                acc_ops.linear,
                kwargs={"input": a, "weight": weight_t_attr, "bias": bias},
            )
        add_node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def trt_transposed_matmul(
    lhs: torch.Tensor, rhs: torch.Tensor, lhs_transposed: bool, rhs_transposed: bool
):
    if lhs_transposed:
        lhs = lhs.transpose(-1, -2)
    if rhs_transposed:
        rhs = rhs.transpose(-1, -2)
    return torch.matmul(lhs, rhs)


def trt_transposed_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
):
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def check_permute(node: torch.fx.Node):
    ranks = len(node.meta["tensor_meta"].shape)
    permutation = list(i % ranks for i in node.kwargs["permutation"])  # type: ignore[union-attr]
    allowed_permutation = list(i for i in range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


@observable()
@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def fuse_permute_linear(gm: torch.fx.GraphModule, input: Input):
    """
    Fuse pattern like permute + linear if permute is transposing the last two dimension.
    """
    for node in gm.graph.nodes:
        if node.target == acc_ops.linear:
            inp = node.kwargs["input"]
            if inp.target == acc_ops.permute and check_permute(inp):
                inp = inp.kwargs["input"]
                weight = node.kwargs["weight"]
                bias = node.kwargs["bias"]
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_function(
                        trt_transposed_linear, args=(inp, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


@observable()
@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def fuse_permute_matmul(gm: torch.fx.GraphModule, input: Input):
    """
    Fuse pattern like permute + matmul if permute is transposing the last two dimension.
    """
    for node in gm.graph.nodes:
        if node.target == acc_ops.matmul:
            lhs, rhs = node.kwargs["input"], node.kwargs["other"]
            lhs_transposed = rhs_tranposed = False
            skip = False

            if lhs.target == acc_ops.permute and check_permute(lhs):
                lhs_transposed = True
                lhs = lhs.kwargs["input"]

            if rhs.target == acc_ops.permute and check_permute(rhs):
                rhs_tranposed = True
                rhs = rhs.kwargs["input"]

            if (not skip) and (lhs_transposed or rhs_tranposed):
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_function(
                        trt_transposed_matmul,
                        args=(lhs, rhs, lhs_transposed, rhs_tranposed),
                    )
                node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def slice_list(sli: slice, dim: int, size: int):
    slice_all = slice(None, None, None)
    if size == 1:
        return [sli]
    elif size == 2:
        if dim == 0:
            return [sli, slice_all]
        elif dim == 1:
            return [slice_all, sli]
    elif size == 3:
        if dim == 0:
            return [sli, slice_all, slice_all]
        elif dim == 1:
            return [slice_all, sli, slice_all]
        elif dim == 2:
            return [slice_all, slice_all, sli]
    elif size == 4:
        if dim == 0:
            return [sli, slice_all, slice_all, slice_all]
        elif dim == 1:
            return [slice_all, sli, slice_all, slice_all]
        elif dim == 2:
            return [slice_all, slice_all, sli, slice_all]
        elif dim == 3:
            return [slice_all, slice_all, slice_all, sli]


def split_across(
    gm: torch.fx.GraphModule, sli: slice, input_node: torch.fx.Node, dim: int, size: int
):
    start_node = end_node = mid_node = None
    if sli.start is None and sli.stop is None:
        return (start_node, input_node, end_node)
    if sli.start is not None:
        st_sli = slice(0, sli.start, None)
        slice_list_gen = slice_list(st_sli, dim, size)
        start_node = gm.graph.call_function(
            operator.getitem, args=(input_node, slice_list_gen)
        )
    if sli.stop is not None:
        end_sli = slice(sli.stop, None, None)
        slice_list_gen = slice_list(end_sli, dim, size)
        end_node = gm.graph.call_function(
            operator.getitem, args=(input_node, slice_list_gen)
        )
    if dim != size - 1:
        mid_sli = slice(sli.start, sli.stop, None)
        slice_list_gen = slice_list(mid_sli, dim, size)
        mid_node = gm.graph.call_function(
            operator.getitem, args=(input_node, slice_list_gen)
        )
    return (start_node, mid_node, end_node)


def list_gen(
    start_node: torch.fx.Node,
    end_node: torch.fx.Node,
    input_node: torch.fx.Node,
    gm: torch.fx.GraphModule,
    dim: int,
):
    if start_node:
        if end_node:
            concat_list = [start_node, input_node, end_node]
        else:
            concat_list = [start_node, input_node]
    else:
        if end_node:
            concat_list = [input_node, end_node]
        else:
            concat_list = [input_node]
    if len(concat_list) > 1:
        concat_node = gm.graph.call_function(torch.cat, args=(concat_list, dim))
    else:
        concat_node = concat_list[0]
    return concat_node


def transform_setitem(gm: torch.fx.GraphModule, input: Input):
    """
    Setitem is not tracable in fx and acc tracer but is available in dynamo trace. This pass works for dynamo trace only.
    The implementation decompose the setitem into a few getitem op and assembly together again through concat.
    The major reason is that TRT does not support in-place copy and memory reference.
    """
    map_replace = {}
    for node in gm.graph.nodes:
        for old_node in map_replace:
            node.replace_input_with(old_node, map_replace[old_node])

        if node.target == operator.setitem:
            input_node = node.args[0]
            sli = node.args[1]
            inp = node.args[2]

            inp_flag = False
            if type(inp) == torch.fx.node.Node and inp.target == operator.getitem:
                new_args = list(copy.deepcopy(inp.args[1]))
                for ind, val in enumerate(new_args):
                    if type(val) == int:
                        inp_flag = True
                        if val == -1:
                            new_args[ind] = slice(-1, None, None)
                        else:
                            new_args[ind] = slice(val, val + 1, None)

                if inp_flag:
                    with gm.graph.inserting_before(inp):
                        new_node = gm.graph.call_function(
                            operator.getitem, args=(inp.args[0], new_args)
                        )
                        inp.replace_all_uses_with(new_node)
                    inp = new_node

            if type(sli) is not tuple:
                sli = [sli]

            tmp_sli = []
            for x in sli:
                if type(x) == int:
                    if x == -1:
                        tmp_sli.append(slice(-1, None, None))
                    else:
                        tmp_sli.append(slice(x, x + 1, None))
                else:
                    tmp_sli.append(x)
            sli = tmp_sli

            dimension = len(sli)
            with gm.graph.inserting_before(node):
                if dimension == 1:
                    start_node_0, _, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=1
                    )
                    concat_node_0 = list_gen(start_node_0, end_node_0, inp, gm, 0)
                elif dimension == 2:
                    start_node_0, mid_node_0, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=2
                    )
                    start_node_1, _, end_node_1 = split_across(
                        gm, sli[1], mid_node_0, dim=1, size=2
                    )
                    concat_node_1 = list_gen(start_node_1, end_node_1, inp, gm, 1)
                    concat_node_0 = list_gen(
                        start_node_0, end_node_0, concat_node_1, gm, 0
                    )
                elif dimension == 3:
                    start_node_0, mid_node_0, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=3
                    )
                    start_node_1, mid_node_1, end_node_1 = split_across(
                        gm, sli[1], mid_node_0, dim=1, size=3
                    )
                    start_node_2, _, end_node_2 = split_across(
                        gm, sli[2], mid_node_1, dim=2, size=3
                    )
                    concat_node_2 = list_gen(start_node_2, end_node_2, inp, gm, 2)
                    concat_node_1 = list_gen(
                        start_node_1, end_node_1, concat_node_2, gm, 1
                    )
                    concat_node_0 = list_gen(
                        start_node_0, end_node_0, concat_node_1, gm, 0
                    )
                elif dimension == 4:
                    start_node_0, mid_node_0, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=4
                    )
                    start_node_1, mid_node_1, end_node_1 = split_across(
                        gm, sli[1], mid_node_0, dim=1, size=4
                    )
                    start_node_2, mid_node_2, end_node_2 = split_across(
                        gm, sli[2], mid_node_1, dim=2, size=4
                    )
                    start_node_3, _, end_node_3 = split_across(
                        gm, sli[3], mid_node_2, dim=3, size=4
                    )
                    concat_node_3 = list_gen(start_node_3, end_node_3, inp, gm, 3)
                    concat_node_2 = list_gen(
                        start_node_2, end_node_2, concat_node_3, gm, 2
                    )
                    concat_node_1 = list_gen(
                        start_node_1, end_node_1, concat_node_2, gm, 1
                    )
                    concat_node_0 = list_gen(
                        start_node_0, end_node_0, concat_node_1, gm, 0
                    )
                else:
                    warnings.warn(f"setitem does not support dimension={dimension}")
                    continue
                node.replace_all_uses_with(concat_node_0)
                map_replace[input_node] = concat_node_0
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def fix_reshape_batch_dim(mod: fx.GraphModule) -> fx.GraphModule:
    """\
    TRT cannot reason about shape patterns like x.reshape(y.size(0), -1, 256),
    since the dynamic shape of the reshape comes from the dynamic shape of
    another node (y). The compilation will fail with various memory related
    errors, depending on the size of the input tensor.

    This pass fixes the issue by finding this reshape pattern, checking that:

        x.size(0) == y.size(0)

    And then replaces reshape's batch size from y.size(0) to x.size(0).
    """

    def get_reshape_batch_size_as_node(maybe_reshape: fx.Node) -> Optional[fx.Node]:
        """\
        Try to find the reshape op's batch size as an input node.

        Match below graph structure and return `node_y`:
            node_x.reshape({"acc_out_ty": {"shape": (node_y, ...)}})
        """
        if (
            maybe_reshape.op != "call_function"
            or maybe_reshape.target != acc_ops.reshape
        ):
            return None
        shape = getattr(maybe_reshape.kwargs["acc_out_ty"], "shape", None)
        if not shape:
            return None
        batch_size = shape[0]
        if isinstance(batch_size, fx.Node):
            return batch_size
        return None

    def get_reshape_batch_size_inferred_source(
        batch_size_node: fx.Node,
    ) -> Optional[fx.Node]:
        """\
        Given a node representing the batch size used for reshape op, we want
        to know if it is coming from below pattern:

            batch_size_node = src.size()[0]

        or in IR graph:

            src -> size(input=_) -> getitem(input=_, idx=0)
                                        ^ ~~~  batch_size_node

        If so, return `src`. Otherwise, return `None`.
        """
        if (
            batch_size_node.op != "call_function"
            or batch_size_node.target != acc_ops.getitem
            or batch_size_node.kwargs["idx"] != 0
        ):
            return None
        maybe_size: fx.Node = batch_size_node.all_input_nodes[0]
        if maybe_size.op != "call_function" or maybe_size.target != acc_ops.size:
            return None
        return maybe_size.all_input_nodes[0]

    maybe_reshape: fx.Node
    for maybe_reshape in mod.graph.nodes:
        reshape_batch_size: Optional[fx.Node] = get_reshape_batch_size_as_node(
            maybe_reshape
        )
        if not reshape_batch_size:
            continue
        reshape_batch_size_inferred_source: Optional[fx.Node] = (
            get_reshape_batch_size_inferred_source(reshape_batch_size)
        )
        if not reshape_batch_size_inferred_source:
            continue

        reshape_input: fx.Node = maybe_reshape.kwargs["input"]
        if reshape_input == reshape_batch_size_inferred_source:
            continue

        if not _is_batch_size_equal(reshape_input, reshape_batch_size_inferred_source):
            continue

        _LOGGER.info(
            f"{fix_reshape_batch_dim}: Found bad pattern:  y.reshape((x, ...)). Reshape node: {maybe_reshape}"
        )

        # Step 1: create a node to compute batch size, using the tensor which
        # is being reshaped: reshape_input.size()[0]. This batch size is now
        # derived from reshape_input, the same node as the reshape op's input.
        with mod.graph.inserting_before(maybe_reshape):
            reshape_batch_size_2: fx.Node = maybe_reshape.graph.call_function(
                acc_ops.getitem,
                kwargs={
                    "idx": 0,
                    "input": maybe_reshape.graph.call_function(
                        acc_ops.size,
                        kwargs={
                            "input": reshape_input,
                        },
                    ),
                },
            )

        # Step 2: update `maybe_reshape`'s shape argument to be
        # (reshape_batch_size_2, *DONT_CARE_JUST_COPY_OVER)
        maybe_reshape.kwargs = {
            **maybe_reshape.kwargs,
            "acc_out_ty": acc_utils.build_raw_tensor_meta(
                shape=(
                    reshape_batch_size_2,
                    *(maybe_reshape.kwargs["acc_out_ty"].shape[1:]),
                )
            ),
        }

    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod


def _is_batch_size_equal(x: fx.Node, y: fx.Node) -> bool:
    """\
    Check that x.size(0) == y.size(0)
    """
    x_size, y_size = _get_shape(x), _get_shape(y)
    return (
        x_size
        and y_size
        # now both are non-empty
        and x_size[0] == y_size[0]
    )


def _get_shape(node: fx.Node) -> Optional[torch.Size]:
    if (
        not getattr(node, "meta", None)
        or not node.meta.get("tensor_meta", None)
        or not getattr(node.meta["tensor_meta"], "shape", None)
    ):
        # shape info not available
        return None
    return node.meta["tensor_meta"].shape


@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def fix_clamp_numerical_limits_to_fp16(
    mod: torch.fx.GraphModule, input: Input
) -> torch.fx.GraphModule:
    MIN_FP16 = -65504.0
    MAX_FP16 = 65504.0
    for node in mod.graph.nodes:
        if node.op == "call_function" and "clamp" in str(node.target):
            input_kwargs = node.kwargs
            if input_kwargs["min"] < MIN_FP16 and input_kwargs["max"] > MAX_FP16:
                new_kwargs = {
                    "input": input_kwargs["input"],
                    "min": MIN_FP16,
                    "max": MAX_FP16,
                }
                node.kwargs = new_kwargs

    mod.recompile()
    return mod


@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def remove_dtype_and_to_pattern(
    mod: torch.fx.GraphModule, input: Input
) -> torch.fx.GraphModule:
    """
    Remove this pattern since it is unnecessary to cast to dtype
        %dtype : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.dtype](args = (), kwargs = {input: %_attention_layers_0__uva})
        %to_18 : [#users=2] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.to_dtype](args = (), kwargs = {input: %x})
    """
    for node in mod.graph.nodes:
        if node.op == "call_function" and node.target == acc_ops.dtype:
            # find its first user
            next_node = next(iter(node.users))
            # acc_op or pt op is treated differently
            input = (
                next_node.kwargs["input"]
                if "input" in next_node.kwargs
                else next_node.args[0]
            )
            if len(node.users) == 1 and (
                next_node.target == acc_ops.to_dtype or next_node.target == "to"
            ):
                next_node.replace_all_uses_with(input)
                mod.graph.erase_node(next_node)
                mod.graph.erase_node(node)

    mod.graph.eliminate_dead_code()
    mod.recompile()
    return mod
