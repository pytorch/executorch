# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from executorch.backends.qualcomm._passes.utils import find_pattern
from executorch.backends.qualcomm.utils.constants import (
    QCOM_BLOCK_SIZE,
    QCOM_QUANT_ATTRS,
    QCOM_REQUANTIZE,
    QCOM_SCALE,
    QCOM_SCALES,
    QCOM_ZERO_POINT,
    QCOM_ZERO_POINTS,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from executorch.exir.passes.constant_prop_pass import constant_prop_pass

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def _is_node(node):
    return isinstance(node, torch.fx.Node)


def _is_output(node):
    return _is_node(node) and node.op == "output"


def _is_call(node):
    return _is_node(node) and node.op == "call_function"


def _is_unsqueeze(node):
    return _is_call(node) and node.target == exir_ops.edge.aten.unsqueeze_copy.default


def _is_view(node):
    return _is_call(node) and node.target == exir_ops.edge.aten.view_copy.default


def _is_permute(node):
    return _is_call(node) and node.target == exir_ops.edge.aten.permute_copy.default


def _is_matmul(node):
    return _is_call(node) and node.target == exir_ops.edge.aten.matmul.default


def _is_bmm(node):
    return _is_call(node) and node.target == exir_ops.edge.aten.bmm.default


def _is_expand(node):
    return _is_call(node) and node.target == exir_ops.edge.aten.expand_copy.default


def _is_conv(node):
    return _is_call(node) and node.target == exir_ops.edge.aten.convolution.default


def _is_softmax(node):
    return _is_call(node) and node.target in [
        exir_ops.edge.aten._softmax.default,
        exir_ops.edge.aten._safe_softmax.default,
    ]


def _shape(node):
    assert "val" in node.meta
    return list(node.meta["val"].shape)


@dataclass
class Sha:
    axis: int
    heads: int

    def __repr__(self):
        return f"Sha(axis={self.axis}, heads={self.heads})"


class ConvertMhaToSha(ExportPass):
    """
    b=batch, e=emb=h*d, h=heads, d=head_size, s=seq_len, p=past, c=s+p

    i[bse] ─┬─ q[bse] ─ [bhsd] ─ RoPE ─ [bhsd] ───────────────────── qk[bhsc] ─ mask ─ softmax ─ qkv[bhsd] ─ [bse] ─ o[bse]
            ├─ k[bse] ─ [bhsd] ─ RoPE ─ [bhds] ─ k_cat[bhdc] ─(k_exp)─┘                           │
            │                     past_k[bhdp] ──┘                                                │
            └─ v[bse] ─ [bhsd] ───────────────── v_cat[bhcd] ─(v_exp)-────────────────────────────┘
                                  past_v[bhpd] ──┘
    """

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        verbose=False,
    ):
        super().__init__()
        self.edge_program = edge_program
        self.verbose = verbose

    def _nodes(self, graph_module, wanted_sources, node_checker=None):
        nodes = []
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in wanted_sources:
                if node_checker is None or node_checker(node):
                    nodes.append(node)
        return nodes

    def _get_attention_output(self, softmax):
        """Output of MHA block or input of output projection"""

        pattern_qk = [_is_softmax, "*", lambda x: _is_matmul(x) or _is_bmm(x)]
        qk = find_pattern(softmax, pattern_qk)
        if not qk:
            return None, None, None

        patterns_qkv = [
            _is_softmax,
            "*",
            lambda x: _is_matmul(x) or _is_bmm(x),
            "*",
            _is_permute,
            _is_view,
        ]

        qkv = find_pattern(softmax, patterns_qkv, from_args=False)
        if qkv is None:
            return None, None, None

        permute, reshape = qkv[0][-2:]
        matmul = qkv[0][2]
        attn_output = matmul
        sha_axis = 1
        remove_nodes = [permute]
        # the shape of attn_output should be [bhsd]
        shape = _shape(attn_output.args[0])
        heads = shape[sha_axis]
        sha = Sha(axis=sha_axis, heads=heads)

        return attn_output, sha, remove_nodes

    def _update_requantize_user(self, node):
        if QCOM_REQUANTIZE in node.meta:
            user_node_list = [user.name for user in node.users.keys()]

            new_dict = {}
            for original_key in node.meta[QCOM_REQUANTIZE]:
                for new_key in user_node_list:
                    # new_keys are the name of the split nodes whose naming pattern follows: <original_key>_h_xxx
                    if original_key in new_key:
                        new_dict.update(
                            {new_key: node.meta[QCOM_REQUANTIZE][original_key]}
                        )
            node.meta[QCOM_REQUANTIZE] = new_dict

    def _split(  # noqa: C901
        self,
        graph_module: torch.fx.GraphModule,
        attn_output: torch.fx.Node,
        sha: Sha,
        remove_nodes: List,
    ):
        """
        Main MHA to SHAs
        - Start from the attention output or the input of the output projection node, assuming the head axis is 2.
        - Recursively visit parent nodes until reaching the static Linear/Conv2D nodes, which must be the Q/K/V projection nodes.
        - Splitting begins from the end of the recursion, which must be the Q/K/V projection nodes.
        - The visit call will return the split nodes, which will be used by subsequent child visitors.

        Known issue
        - Packed Q/K/V projection is not supported yet
        """

        def _visit_reshape(node, sha):
            """Reshape: handle GQA pattern"""
            in_shape, out_shape = _shape(node.args[0]), _shape(node)
            if out_shape[sha.axis] % sha.heads == 1:
                return _no_split(node, sha)

            assert (
                out_shape[sha.axis] % sha.heads == 0
            ), f"mismatching num_heads, {out_shape[sha.axis]} % {sha.heads} != 0"

            pattern_simple_gqa = [
                _is_view,
                lambda x: _is_expand(x) and len(_shape(x)) == 5,
                _is_unsqueeze,
            ]

            if gqa := find_pattern(node, pattern_simple_gqa):
                # GQA pattern: skip these and adjust sha.heads
                if self.verbose:
                    logging.info(f"{__name__}:_visit_reshape: {node} is for GQA!")
                _, expand, unsqueeze = gqa[0]
                expand_shape = expand.args[1]
                unsqueeze_dim = unsqueeze.args[1]
                repeat_count = expand_shape[unsqueeze_dim]
                kv_sha = Sha(sha.axis, in_shape[sha.axis])
                new_arg0s = _visit(unsqueeze.args[0], kv_sha)
                new_arg0s = [arg for arg in new_arg0s for _ in range(repeat_count)]
            else:
                new_arg0s = _visit(node.args[0], sha)

            out_shape[sha.axis] //= sha.heads
            new_args = [(arg0, out_shape) for arg0 in new_arg0s]
            return _split_call(node, sha, new_args, out_shape)

        def _visit_permute(node, sha):
            """Transpose: permute sha axis as well"""
            out_shape = _shape(node)
            assert (
                out_shape[sha.axis] % sha.heads == 0
            ), f"mismatching num_heads, {out_shape[sha.axis]} % {sha.heads} != 0"
            out_shape[sha.axis] //= sha.heads
            permute = node.args[1]
            sha_permuted = Sha(axis=permute[sha.axis], heads=sha.heads)
            new_arg0s = _visit(node.args[0], sha_permuted)
            new_args = [(arg0, node.args[1]) for arg0 in new_arg0s]
            return _split_call(node, sha, new_args, out_shape)

        def _visit_expand(node, sha):
            out_shape = _shape(node)
            if out_shape[sha.axis] != 1:
                assert (
                    out_shape[sha.axis] % sha.heads == 0
                ), f"mismatching num_heads, {out_shape[sha.axis]} % {sha.heads} != 0"
                out_shape[sha.axis] //= sha.heads

            exp_shape = node.args[1]
            if exp_shape[sha.axis] == 1:
                return _visit_default(node, sha)

            assert (
                exp_shape[sha.axis] % sha.heads == 0
            ), f"mismatching expand shape, {exp_shape[sha.axis]} % {sha.heads} != 0"
            new_exp_shape = type(exp_shape)(
                [
                    dim // sha.heads if axis == sha.axis else dim
                    for axis, dim in enumerate(exp_shape)
                ]
            )
            new_args = [(node.args[0], new_exp_shape)] * sha.heads
            new_nodes = _split_call(node, sha, new_args, out_shape)
            return new_nodes

        def _visit_cat(node, sha):
            out_shape = _shape(node)
            if out_shape[sha.axis] != 1:
                assert (
                    out_shape[sha.axis] % sha.heads == 0
                ), f"mismatching num_heads, {out_shape[sha.axis]} % {sha.heads} != 0"
                out_shape[sha.axis] //= sha.heads

            assert isinstance(node.args[0], (tuple, list))  # concat
            split_arg0s = [_visit(arg, sha) for arg in node.args[0]]
            new_arg0s = list(zip(*split_arg0s))
            split_arg1s = [_visit(arg, sha) for arg in node.args[1:]]
            new_arg1s = list(zip(*split_arg1s))
            new_args = [(arg0, *arg1) for arg0, arg1 in zip(new_arg0s, new_arg1s)]

            new_nodes = _split_call(node, sha, new_args, out_shape)
            return new_nodes

        def _visit_default(node, sha):
            out_shape = _shape(node)

            if out_shape[sha.axis] != 1:
                assert (
                    out_shape[sha.axis] % sha.heads == 0
                ), f"mismatching num_heads, {out_shape[sha.axis]} % {sha.heads} != 0"
                out_shape[sha.axis] //= sha.heads

            assert not isinstance(
                node.args[0], (tuple, list)
            ), f"Unexpected cat node:{node}"
            split_args = [_visit(arg, sha) for arg in node.args]
            new_args = list(zip(*split_args))
            new_nodes = _split_call(node, sha, new_args, out_shape)
            return new_nodes

        def _is_mha(node, sha):
            if not _is_node(node):
                return False
            out_shape = _shape(node)
            return len(out_shape) > sha.axis and out_shape[sha.axis] == sha.heads

        def _visit_binary(node, sha):
            """elementwise binary operator visit mha inputs only"""
            out_shape = _shape(node)
            if out_shape[sha.axis] != 1:
                assert (
                    out_shape[sha.axis] % sha.heads == 0
                ), f"mismatching num_heads, {out_shape[sha.axis]} % {sha.heads} != 0"
                out_shape[sha.axis] //= sha.heads

            split_args = [
                (_visit(arg, sha) if _is_mha(arg, sha) else [arg] * sha.heads)
                for arg in node.args
            ]
            new_args = list(zip(*split_args))
            new_nodes = _split_call(node, sha, new_args, out_shape)
            return new_nodes

        def _visit_placeholder(node, sha):
            in_shape = _shape(node)
            if (
                in_shape
                and len(in_shape) > sha.axis
                and in_shape[sha.axis] == sha.heads
            ):  # split past_kv by heads
                new_nodes = _split_placeholder(
                    node, axis=sha.axis, size=1, count=sha.heads
                )
            else:
                # position embedding, attention mask, and R3 weights
                new_nodes = _no_split(node, sha)
            return new_nodes

        def _get_slicers(count, axis, size):
            return [
                tuple(
                    [
                        (
                            slice(size * idx, size * (idx + 1))
                            if ax == axis
                            else slice(None)
                        )
                        for ax in range(axis + 1)
                    ]
                )
                for idx in range(count)
            ]

        def _split_call(node, sha, new_args, out_shape):
            with graph_module.graph.inserting_after(node):
                new_nodes = []
                slicers = _get_slicers(sha.heads, sha.axis, out_shape[sha.axis])
                for head, (args, slicer) in enumerate(zip(new_args, slicers)):
                    name = f"{node.name}_h_{head}"
                    new_nodes.append(
                        _duplicate_call(node, args, None, slicer, name=name)
                    )
            return new_nodes

        def _create_call(
            op_target, args: Tuple, kwargs: Optional[dict] = None, name: str = None
        ):
            return graph_module.graph.create_node(
                "call_function",
                op_target,
                args=args,
                kwargs=kwargs or {},
                name=name,
            )

        def _no_split(node, sha):
            return [node] * sha.heads

        def _copy_meta(dst_node, src_node, slicer):
            dst_node.meta = src_node.meta.copy()
            dst, src = dst_node.meta, src_node.meta
            if "val" in src:
                dst["val"] = src["val"].clone()[slicer]
                if src_tensor_meta := src.get("tensor_meta", None) is not None:
                    tensor_meta = dict(zip(src_tensor_meta._fields, [*src_tensor_meta]))
                    tensor_meta["shape"] = dst["val"].shape
                    tensor_meta["stride"] = dst["val"].stride()
                    dst["tensor_meta"] = type(src_tensor_meta)(**tensor_meta)
            # PCQ
            if QCOM_QUANT_ATTRS in src and QCOM_SCALES in src[QCOM_QUANT_ATTRS]:
                dst[QCOM_QUANT_ATTRS] = src[QCOM_QUANT_ATTRS].copy()
                # slice for per channel quantize
                dst[QCOM_QUANT_ATTRS][QCOM_SCALES] = src[QCOM_QUANT_ATTRS][
                    QCOM_SCALES
                ].clone()[slicer]
                dst[QCOM_QUANT_ATTRS][QCOM_ZERO_POINTS] = src[QCOM_QUANT_ATTRS][
                    QCOM_ZERO_POINTS
                ].clone()[slicer]

            # LPBQ
            if QCOM_QUANT_ATTRS in src and QCOM_BLOCK_SIZE in src[QCOM_QUANT_ATTRS]:
                dst[QCOM_QUANT_ATTRS] = src[QCOM_QUANT_ATTRS].copy()
                dst[QCOM_QUANT_ATTRS][QCOM_SCALE] = src[QCOM_QUANT_ATTRS][
                    QCOM_SCALE
                ].clone()[slicer]
                dst[QCOM_QUANT_ATTRS][QCOM_ZERO_POINT] = src[QCOM_QUANT_ATTRS][
                    QCOM_ZERO_POINT
                ].clone()[slicer]

            if "example_value" in src:
                dst["example_value"] = src["example_value"].clone()[slicer]

            if QCOM_REQUANTIZE in src:
                # We assume there is no requantize happens on the per-channel quantization weights, only per-tensor quantization
                dst[QCOM_REQUANTIZE] = src[QCOM_REQUANTIZE].copy()

        def _duplicate_call(
            node, args: Tuple, kwargs: Optional[dict] = None, slicer=None, name=None
        ):
            """Create SHA nodes by duplicating"""
            assert (
                node.op == "call_function"
            ), f"Unexpected node:{node.name}:{node.target}"
            new_node = _create_call(node.target, args, kwargs, name=name)
            _copy_meta(new_node, node, slicer)
            return new_node

        def _split_placeholder(node, axis, size, count):
            slice_op = exir_ops.edge.aten.slice_copy.Tensor
            with graph_module.graph.inserting_after(node):
                sliced_nodes = []
                for head, slicer in zip(range(count), _get_slicers(count, axis, size)):
                    sliced = _create_call(
                        slice_op,
                        (node, axis, slicer[axis].start, slicer[axis].stop),
                        name=f"{node.name}_h_{head}",
                    )
                    _copy_meta(sliced, node, slicer)
                    sliced_nodes.append(sliced)
            return sliced_nodes

        def _visit_linear_conv(node, sha):
            """
            0. Reshape of making multi-heads of MHA
                - embedding = head * head_dim
                - [batch, sequence, embedding] -> [batch, sequence, head, head_dim],
                - [batch, sequence, embedding, 1] -> [batch, sequence, head, head_dim], embedding=head * head_dim

            1. **q/k/v projections => stop recursion**
                - 3D input and output
                - Split output features
                - ConvInplaceLinear
                    - [3d-unsqueeze-4d-permute-conv2d-permute-squeeze-3d]
                    - input: permute_copy(input): 4D[batch, in_feature, 1, num_input] => re-use
                    - weight[out_feature = heads * head_dim, in_feature, 1, 1] => heads * [head_dim, in_feature, 1, 1]
                - So, split_axis=0 for Conv2D

            2. **R3 of SpinQuant => continue recursion**
                - 4D input and output
                - ConvInplaceLinear
                    - [4d-permute-conv2d-permute-4d], **same as 3D case but no squeeze/unsqueeze**
                    - input: 4D [batch, head_dim, heads, num_input] => heads * [batch, head_dim, 1, num_input]
                    - weight: 2D [head_dim, head_dim, 1, 1] => re-use
            """

            def _is_making_mha(cur):
                cur_sha = sha
                pattern_conv_mha = ([_is_conv, "*", _is_permute, "*", _is_view], False)
                if mha := find_pattern(cur, *pattern_conv_mha):
                    permute, reshape = mha[0][-3], mha[0][-1]
                    permutation = permute.args[1]
                    cur_sha = Sha(
                        permutation.index(sha.axis), sha.heads
                    )  # to reverse permute
                else:
                    return False

                # Check whether this reshape is to make multi-heads or not
                if len(reshape.args[1]) == 4:
                    # got MHA reshape
                    in_shape, out_shape = _shape(reshape.args[0]), _shape(reshape)
                    if (
                        len(out_shape) > cur_sha.axis + 1
                        and in_shape[cur_sha.axis]
                        == out_shape[cur_sha.axis] * out_shape[cur_sha.axis + 1]
                    ):
                        return True
                return False

            if _is_making_mha(node):
                if self.verbose:
                    logging.info(
                        f"{__name__}:_visit_linear_conv: {node} is making MHA!"
                    )
                out_feature, *_ = _shape(node.args[1])
                assert out_feature % sha.heads == 0
                out_feature_per_head = out_feature // sha.heads

                split_axis = 0
                new_weights = _split_placeholder(
                    node.args[1],
                    axis=split_axis,
                    size=out_feature_per_head,
                    count=sha.heads,
                )
                if node.args[2] is not None:
                    new_bias = _split_placeholder(
                        node.args[2],
                        axis=split_axis,
                        size=out_feature_per_head,
                        count=sha.heads,
                    )

                with graph_module.graph.inserting_after(node):
                    new_nodes = []
                    slicers = _get_slicers(sha.heads, 1, out_feature_per_head)
                    if node.args[2] is not None:
                        for head, (weight, bias, slicer) in enumerate(
                            zip(new_weights, new_bias, slicers)
                        ):
                            name = f"{node.name}_h_{head}"
                            sliced = _duplicate_call(
                                node,
                                (node.args[0], weight, bias) + node.args[3:],
                                None,
                                slicer,
                                name=name,
                            )
                            new_nodes.append(sliced)
                    else:
                        for head, (weight, slicer) in enumerate(
                            zip(new_weights, slicers)
                        ):
                            name = f"{node.name}_h_{head}"
                            sliced = _duplicate_call(
                                node,
                                (node.args[0], weight) + node.args[2:],
                                None,
                                slicer,
                                name=name,
                            )
                            new_nodes.append(sliced)

                return new_nodes
            else:
                return _visit_default(node, sha)

        def _concat_sha_nodes(node, sha):
            """Concat sha nodes and replace old node"""
            sha_nodes = visited[node]
            with graph_module.graph.inserting_after(sha_nodes[0]):
                cat = exir_ops.edge.aten.cat.default
                name = f"{node.name}_sha_concat"
                new_node = _create_call(cat, (sha_nodes, sha.axis), name=name)
                new_node.meta = node.meta.copy()
                fake_tensors = [n.meta["val"] for n in sha_nodes]
                result_fake_tensor = torch.cat(fake_tensors, sha.axis)
                new_node.meta["val"] = result_fake_tensor
                node.replace_all_uses_with(new_node)

        def _visit(node, sha):
            if not _is_node(node):
                return [node for _ in range(sha.heads)]

            if node in visited:
                return visited[node]

            visitors = {
                "placeholder": _visit_placeholder,
                exir_ops.edge.aten.expand_copy.default: _visit_expand,
                exir_ops.edge.aten.view_copy.default: _visit_reshape,
                exir_ops.edge.aten.permute_copy.default: _visit_permute,
                exir_ops.edge.aten.convolution.default: _visit_linear_conv,
                exir_ops.edge.aten.mm.default: _visit_linear_conv,
                exir_ops.edge.aten.cat.default: _visit_cat,
                exir_ops.edge.aten.add.Tensor: _visit_binary,
                exir_ops.edge.aten.mul.Tensor: _visit_binary,
                exir_ops.edge.aten.eq.Tensor: _no_split,
            }

            target = node.target if _is_call(node) else node.op
            visited[node] = visitors.get(target, _visit_default)(node, sha)

            if [user for user in node.users.keys() if _is_output(user)]:
                _concat_sha_nodes(node, sha)
            return visited[node]

        if self.verbose:
            logging.info(f"{__name__}:_split: attn_output:{attn_output}, sha:{sha}!")
        visited = {}
        _visit(attn_output, sha)
        opt_sha = Sha(axis=3, heads=sha.heads)
        _concat_sha_nodes(attn_output, opt_sha)
        for remove_node in remove_nodes:
            assert _is_permute(remove_node) or _is_view(
                remove_node
            ), "The removed nodes must be either transpose or reshape"
            rnode_input = remove_node.args[0]
            for user in list(remove_node.users):
                new_args = tuple(
                    rnode_input if arg is remove_node else arg for arg in user.args
                )
                user.args = new_args
        for remove_node in remove_nodes:
            graph_module.graph.erase_node(remove_node)

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        softmaxes = self._nodes(
            graph_module,
            [
                exir_ops.edge.aten._softmax.default,
                exir_ops.edge.aten._safe_softmax.default,
            ],
        )
        for softmax in softmaxes:
            attn_output, sha, remove_nodes = self._get_attention_output(softmax)
            if not attn_output:
                continue

            self._split(graph_module, attn_output, sha, remove_nodes)
            modified = True

        if modified:
            for node in graph_module.graph.nodes:
                self._update_requantize_user(node)
            graph_module.graph.eliminate_dead_code()
            constant_prop_pass(self.edge_program)  # need to fuse sha weights
        graph_module.recompile()
        graph_module.graph.lint()

        return PassResult(graph_module, modified=modified)
