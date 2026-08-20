# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy

import torch
import torch.utils._pytree as pytree
from executorch.exir.delegate import LoweredBackendModule
from executorch.exir.dynamic_shape import (
    calculate_dynamic_shape_spec,
    DynamicMemoryPlanningMode,
)
from executorch.exir.pass_base import Argument, ExportPass
from executorch.exir.pass_infra.node_metadata import NodeMetadata
from executorch.exir.pass_infra.proxy_value import ProxyValue
from executorch.exir.passes.executorch_prim_ops_registry import _EXECUTORCH_SYM_OPS
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.sym_util import collect_free_symbols, eval_expr
from executorch.exir.tensor import TensorSpec
from torch._subclasses import FakeTensor
from torch.fx import GraphModule


@dataclass
class DSInfo:
    """
    Dynamic shape information we are tracking for each dynamic shape symbol.
    """

    # the output of format_node() for the node introducing the symbol
    node_debug_str: str
    # upper bound value or None for fully dynamic memory planning
    ubval: Optional[int]


class DynamicShapePropPass(ExportPass):
    """
    In general, for each op, this pass propagate dynamic shape information from
    op inputs to op outputs.

    For cond/map nodes, we need pass dynamic shape information to submodules'
    placeholder nodes, propagate the dynamic shape information thru the graphs
    of the submodules, and finally set the node's dynamic shape info based on
    submodules' output nodes' dynamic shape info.
    """

    def __init__(
        self, mode: DynamicMemoryPlanningMode = DynamicMemoryPlanningMode.UPPER_BOUND
    ):
        """
        mode controls how we do memory planning for dynamic shape tensors.
        In UPPER_BOUND mode, we plan dynamic shape tensors' memory based on
        its upper bound shape;
        In FULL_DYNAMIC mdoe, the compiler does not allocata memory for
        dynamic shape tensors, the runtime will do the allocation.
        """
        super().__init__()
        self.mode = mode
        self.sym_to_dsinfo = {}
        self.shape_env = None

    @contextmanager
    def apply_upper_bounds(self):
        """
        Context manager to use upper bound value to evaluate expressions.
        """
        try:
            if self.shape_env:
                old_var_to_val = dict(self.shape_env.var_to_val)
                for sym, dsinfo in self.sym_to_dsinfo.items():
                    assert dsinfo.ubval is not None
                    self.shape_env.var_to_val[sym] = sympy.Integer(dsinfo.ubval)
            yield
        finally:
            if self.shape_env:
                self.shape_env.var_to_val = old_var_to_val

    def copy_dsinfo_btw_specs(self, src_spec: TensorSpec, dst_spec: TensorSpec):
        dst_spec.shape_dynamism = src_spec.shape_dynamism
        dst_spec._upper_bound_shape = src_spec._upper_bound_shape

    def inject_dsinfo_to_graph(
        self,
        subgm: GraphModule,
        inputs: Union[List[ProxyValue], Tuple[ProxyValue, ...]],
        ignore_first_ph: bool = False,
    ):
        """
        ignore_first_ph: This argument is added for map node. For map node,
            the first placeholder is special and we need ignore it here
            and handle it specially.
        """
        phs = [n for n in subgm.graph.nodes if n.op == "placeholder"]
        if ignore_first_ph:
            phs = phs[1:]
        assert len(phs) == len(inputs)
        for ph, inp in zip(phs, inputs):
            dst_spec: TensorSpec = ph.meta["spec"]
            src_spec: TensorSpec = inp.node.meta["spec"]
            self.copy_dsinfo_btw_specs(src_spec, dst_spec)

    def inject_xs_dsinfo_to_graph(self, subgm: GraphModule, xs: ProxyValue):
        """
        xs means the first argument for the map node.

        Even if xs is a upper bound tensor, it's possible that the first placeholder
        of subgm is still a static shape tensor if only the first dimension of xs
        is dynamic. But we don't have this optimization yet. If xs is dynamic,
        we treat the first placeholder of subgm as dynamic.
        """
        ph = next(n for n in subgm.graph.nodes if n.op == "placeholder")
        src_spec: TensorSpec = xs.node.meta["spec"]
        dst_spec = ph.meta["spec"]

        self.copy_dsinfo_btw_specs(src_spec, dst_spec)
        # update dst_spec to remove the highest dimesion
        if dst_spec._upper_bound_shape:
            dst_spec._upper_bound_shape = dst_spec._upper_bound_shape[1:]

    def verify_dsinfo_from_both_branches(
        self, true_gm: GraphModule, false_gm: GraphModule
    ):
        """
        For cond node, true and false branch should return outputs with the
        same shape.
        """
        *_, true_out = true_gm.graph.nodes
        *_, false_out = false_gm.graph.nodes
        true_out = pytree.tree_flatten(true_out)[0]
        false_out = pytree.tree_flatten(false_out)[0]
        assert len(true_out) == len(false_out)
        for true_out_item, false_out_item in zip(true_out, false_out):
            true_spec = true_out_item.meta["spec"]
            false_spec = false_out_item.meta["spec"]
            assert true_spec.shape_dynamism == false_spec.shape_dynamism
            assert true_spec._upper_bound_shape == false_spec._upper_bound_shape

    def extract_dsinfo_from_graph(self, subgm: GraphModule, meta: NodeMetadata):
        *_, out_node = subgm.graph.nodes
        dst_spec_list = pytree.tree_flatten(meta["spec"])[0]
        src_spec_list = pytree.tree_flatten(out_node.meta["spec"])[0]
        for src_spec, dst_spec in zip(src_spec_list, dst_spec_list):
            self.copy_dsinfo_btw_specs(src_spec, dst_spec)

    def call_cond(
        self,
        pred: ProxyValue,
        true_fn: torch.fx.GraphModule,
        false_fn: torch.fx.GraphModule,
        inputs: List[Any],
        meta: NodeMetadata,
    ) -> ProxyValue:
        self.inject_dsinfo_to_graph(true_fn, inputs)
        self.inject_dsinfo_to_graph(false_fn, inputs)
        retval = super().call_cond(pred, true_fn, false_fn, inputs, meta)

        self.verify_dsinfo_from_both_branches(true_fn, false_fn)

        # Note: 'meta' will override the metadata in retval.
        # so we update 'meta' rather than 'retval' here.
        self.extract_dsinfo_from_graph(true_fn, meta)
        return retval

    def call_map(
        self,
        f: torch.fx.GraphModule,
        xs: ProxyValue,
        args: Tuple[ProxyValue, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        self.inject_dsinfo_to_graph(f, args, True)
        self.inject_xs_dsinfo_to_graph(f, xs)
        retval = super().call_map(f, xs, args, meta)

        # We are being a bit conservative that if xs of f's output are dynamic
        # shape, we decide the output of map node as dynamic shape.
        xs_spec = xs.node.meta["spec"]
        *_, subgm_out = f.graph.nodes
        subgm_out_spec = subgm_out.meta["spec"]

        # Take advantage that the static TensorShapeDynamsim is miminal
        result_spec = meta["spec"]
        result_spec.shape_dynamism = max(
            spec.shape_dynamism
            for spec in pytree.tree_flatten((xs_spec, subgm_out_spec))[0]
        )
        if result_spec.shape_dynamism == TensorShapeDynamism.DYNAMIC_BOUND:
            # on the right hand side of the assignment we use 'upper_bound_shape'
            # rather than '_upper_bound_shape'. The former return the static shape
            # for static tensor which is what we want.
            result_spec._upper_bound_shape = (
                xs_spec.upper_bound_shape[:1] + subgm_out_spec.upper_bound_shape
            )

        return retval

    def add_symint_upperbound(
        self, node_debug_str: str, symint: torch.SymInt, ubval: int
    ):
        if not isinstance(symint, torch.SymInt):
            return
        expr = symint.node.expr
        if isinstance(expr, sympy.Symbol):
            self.sym_to_dsinfo[expr] = DSInfo(node_debug_str, ubval)
            if self.shape_env is None:
                self.shape_env = symint.node.shape_env
            else:
                assert symint.node.shape_env is self.shape_env

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        output = super().placeholder(name, arg, meta)
        # TODO: handle full dynamic
        if (
            self.mode == DynamicMemoryPlanningMode.UPPER_BOUND
            and meta.data.get("spec", None) is not None
            and meta.data.get("val", None) is not None
        ):
            spec = meta.data["spec"]
            val = meta.data["val"]
            if not isinstance(val, FakeTensor):
                return output

            if spec.shape_dynamism != TensorShapeDynamism.DYNAMIC_BOUND:
                return output

            for sym, ubval in zip(val.shape, spec._upper_bound_shape):
                assert self.node_debug_str is not None
                self.add_symint_upperbound(self.node_debug_str, sym, ubval)
        return output

    def eval_symint_to_ubval(self, symint: torch.SymInt) -> int:
        return eval_expr(symint)

    def decide_upper_bound_from_symbols(self, meta):
        with self.apply_upper_bounds():
            meta = meta.data
            if meta.get("val", None) is None or meta.get("spec", None) is None:
                return
            vallist, _ = pytree.tree_flatten(meta["val"])
            speclist, _ = pytree.tree_flatten(meta["spec"])
            for val, spec in zip(vallist, speclist):
                if not isinstance(val, FakeTensor) or not isinstance(spec, TensorSpec):
                    continue
                free_symbols = collect_free_symbols(val.shape)
                if len(free_symbols & set(self.sym_to_dsinfo.keys())) == 0:
                    spec.shape_dynamism = TensorShapeDynamism.STATIC
                    spec._upper_bound_shape = None
                    continue
                spec.shape_dynamism = TensorShapeDynamism.DYNAMIC_BOUND
                # evaluate the upper bound shape
                spec._upper_bound_shape = [
                    self.eval_symint_to_ubval(s) for s in val.shape
                ]

    def call_delegate(
        self,
        lowered_module: LoweredBackendModule,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        """
        Override this method so we can properly calculate the dynamic shape
        information for the output of delegate.
        """
        if self.mode == DynamicMemoryPlanningMode.UPPER_BOUND:
            self.decide_upper_bound_from_symbols(meta)
        else:
            raise RuntimeError("NYI: delegatoin supporting in full dynamic mode")
        return super().call_delegate(lowered_module, args, kwargs, meta)

    def call_operator(self, op, args, kwargs, meta):
        """
        If any of the arguments has dynamic shape, mark the output as dynamic shape.
        """

        # no need to do dynamic shape propagation for these ops
        if op.target in _EXECUTORCH_SYM_OPS:
            return super().call_operator(op, args, kwargs, meta)

        if self.mode == DynamicMemoryPlanningMode.UPPER_BOUND:
            self.decide_upper_bound_from_symbols(meta)
            return super().call_operator(op, args, kwargs, meta)

        ds_spec = calculate_dynamic_shape_spec(self.mode, op.target, args, kwargs)

        out_tensor_spec = meta["spec"]

        for ds_spec_item, tensor_spec_item in zip(
            pytree.tree_flatten(ds_spec)[0], pytree.tree_flatten(out_tensor_spec)[0]
        ):
            tensor_spec_item.shape_dynamism = ds_spec_item.shape_dynamism
            tensor_spec_item._upper_bound_shape = ds_spec_item.upper_bound_shape
        return super().call_operator(op, args, kwargs, meta)
