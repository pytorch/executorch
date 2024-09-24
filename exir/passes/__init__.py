# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import _operator
import copy
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from executorch.exir import control_flow, memory, memory_planning
from executorch.exir.common import override_logger
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects.backend._ops import BackendOpOverload
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode
from executorch.exir.error import InternalError
from executorch.exir.operator.convert import (
    get_out_args_from_opoverload,
    is_out_variant,
    to_out_variant,
    to_scratch_op,
)

from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes.const_prop_pass import ConstPropPass
from executorch.exir.passes.debug_handle_generator_pass import DebugHandleGeneratorPass

from executorch.exir.passes.executorch_prim_ops_registry import _EXECUTORCH_SYM_OPS
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    insert_write_back_for_buffers_pass,
)
from executorch.exir.passes.memory_format_ops_pass import MemoryFormatOpsPass
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.normalize_transpose_pass import NormalizeTransposePass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.remove_noop_pass import RemoveNoopPass, RemoveToCopyPass
from executorch.exir.passes.replace_aten_with_edge_pass import OpReplacePass
from executorch.exir.passes.replace_broken_ops_with_function_ops_pass import (
    ReplaceBrokenOpsWithFunctionalOpsPass,
)
from executorch.exir.passes.replace_edge_with_backend_pass import EdgeToBackendOpsPass
from executorch.exir.passes.replace_sym_size_op_pass import ReplaceSymSizeOpPass
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.passes.sym_shape_eval_pass import HintBasedSymShapeEvalPass
from executorch.exir.passes.sym_to_tensor_pass import SymToTensorPass
from executorch.exir.passes.weights_to_outputs_pass import weights_to_outputs_pass
from torch import fx
from torch._subclasses import FakeTensor
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import TensorMetadata

__all__ = [
    "ExportPass",
    "ConstPropPass",
    "QuantFusionPass",
    "OpReplacePass",
    "EdgeToBackendOpsPass",
    "MemoryFormatOpsPass",
    "MemoryPlanningPass",
    "HintBasedSymShapeEvalPass",
    "insert_write_back_for_buffers_pass",
    "weights_to_outputs_pass",
]

Argument = Optional[
    Union[
        Tuple["Argument", ...],
        List["Argument"],
        Dict[str, "Argument"],
        slice,
        torch.fx.Node,
        str,
        int,
        float,
        bool,
        complex,
        torch.dtype,
        torch.Tensor,
        torch.device,
        torch.memory_format,
        torch.layout,
    ]
]


def update_args(
    args: Tuple[Argument, ...], key: int, val: torch.fx.Node
) -> Tuple[Argument, ...]:
    """
    A helper function to update an argument container without changing it.
    This can be used with both args and kwargs.
    """
    if isinstance(args, dict):
        new_dict = copy.copy(args)
        new_dict[key] = val
        return new_dict

    assert isinstance(args, tuple)
    new_tuple = list(args)
    new_tuple[key] = val
    return tuple(new_tuple)


class DebugPass(PassBase):
    def __init__(
        self,
        msg: str = "",
        enable_debug_pass: bool = True,
        show_src: bool = False,
        show_full_path: bool = False,
        show_all_frames: bool = False,
        path_filter: Optional[str] = None,
        show_spec: bool = False,
        log_filename: Optional[str] = None,
    ) -> None:
        """
        show_src: whether to show source code that generated each fx Node
        show_full_path: whether to show the full path of source code or just the filename
        show_all_frames: control for each node whether show only the last frame or all the frames.
        path_filter: a regular expression to filter the path of the stackframes
        log_filename: if provided, the output will also be written to this path.
            Existing content in this file will be discarded.
        """
        self.msg = msg
        self.enable_debug_pass = enable_debug_pass
        self.show_src = show_src
        self.show_full_path = show_full_path
        self.show_all_frames = show_all_frames
        self.show_spec = show_spec
        self.log_filename = log_filename
        if path_filter:
            self.path_filter_re = re.compile(path_filter)  # pyre-ignore
        else:
            self.path_filter_re = None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """
        Counts the number of operations and call_funciton operations.
        """
        if not self.enable_debug_pass:
            return PassResult(graph_module, False)
        # it doesn't make sense to mute the DebugPass if user already
        # specify self.enable_debug_pass to be true
        with override_logger(filename=self.log_filename):
            self.callWithLoggerEnabled(graph_module)
        return PassResult(graph_module, True)

    def printFrames(self, node: fx.Node) -> None:
        """
        The DebugPass maybe used for graph generated by both the old exir dispatch
        tracer or the new pt2 tracer.
        The former store 'stack_trace' field as a json string;
        the latter store 'stack_trace' field as a free form string like:
          ```
            File "/data/sandcastle/boxes/fbsource/buck-out/v2/gen/fbcode/20c706e99f51cf3a/executorch/test/end2end/__end2end__/end2end#link-tree/executorch/test/end2end/test_end2end.py", line 150, in forward
                o = o * a
          ```
        Make this method handle both format. In future, maybe we can drop the
        support for old exir dispatch tracer.
        """
        if (
            self.show_src
            and "stack_trace" in node.meta
            and len(node.meta["stack_trace"]) > 0
        ):
            try:
                stack_trace = json.loads(node.meta["stack_trace"])
                is_json = True
            except json.decoder.JSONDecodeError:
                is_json = False

            if not is_json:
                logging.debug(node.meta["stack_trace"])
                return

            frame_list = []  # tuple of filename, frame name, line number and line
            for frame in stack_trace:
                filename = frame["filename"]
                name = frame["name"]
                lineno = frame["lineno"]
                line = frame["line"]
                if not self.show_full_path:
                    filename = os.path.basename(filename)
                mark = "#link-tree/"
                if mark in filename:
                    filename = filename.split(mark)[-1]

                if not self.path_filter_re or self.path_filter_re.search(filename):
                    frame_list.append((filename, name, lineno, line))

            if not self.show_all_frames:
                frame_list = frame_list[-1:]
            for filename, name, lineno, line in frame_list:
                logging.debug(f"      > {filename}:{lineno} in {name}: {line}")

    def callWithLoggerEnabled(self, graph_module: torch.fx.GraphModule) -> None:
        if self.msg:
            logging.debug(self.msg)
        logging.debug("Enter debug_pass")
        graph_module.recompile()
        logging.debug(f"Code is:\n{graph_module.code}")
        op_to_cnt = defaultdict(int)  # stats for op type
        func_to_cnt = defaultdict(int)  # stats for targets in call_function type
        logging.debug("Nodes:")
        idx = 0
        for node in graph_module.graph.nodes:
            # TODO: better to print python code along with TensorSpecs
            logging.debug(f"{idx:4}: {node.format_node()}")
            if self.show_spec:
                specs = memory_planning.get_node_tensor_specs(node)
                for spec in specs:
                    logging.debug(f"      {spec.debug()}")
                logging.debug(f"      val: {node.meta.get('val', None)}")
            self.printFrames(node)
            idx += 1
            op_to_cnt[node.op] += 1

            if node.op == "call_function":
                target = str(node.target)
                func_to_cnt[target] += 1

        logging.debug("-- node op type stat --")
        for op, cnt in op_to_cnt.items():
            logging.debug(f" op {op}, cnt {cnt}")

        logging.debug("-- call_function stat --")
        for fn, cnt in func_to_cnt.items():
            logging.debug(f" fn {fn}, cnt {cnt}")


# Skip these ops when converting to out variants. They will be handled and
# removed by the emitter.
# pyre-ignore
to_out_var_skiplist: Set[Callable[[Any], Any]] = {
    _operator.getitem,
    torch.ops.higher_order.cond,
    control_flow.while_loop,
    # memory.alloc will be added after the to_out_variant pass so usually
    # we won't see it in the input graph to the to_out_variant pass, unless
    # it's retraced after running to_out_variant with the first trace.
    memory.alloc,
    memory.view,
    executorch_call_delegate,
    torch.ops.aten.copy_.default,
}
to_out_var_skiplist.update(_EXECUTORCH_SYM_OPS)


def make_alloc_node(
    graph_module: torch.fx.GraphModule,
    val: Union[
        Optional[FakeTensor], List[Optional[FakeTensor]], Tuple[Optional[FakeTensor]]
    ],
    tensor_meta: Union[
        Optional[TensorMetadata],
        List[Optional[TensorMetadata]],
        Tuple[Optional[TensorMetadata]],
    ],
) -> torch.fx.Node:
    """
    Note: tensor_metadata is only used in the case of a Tensor subclass, since
    fakifying a tensor subclass is not supported right now
    """
    if val is None:
        if tensor_meta is not None:
            assert isinstance(tensor_meta, TensorMetadata)
            alloc_spec = (tensor_meta.shape, tensor_meta.dtype)
        else:
            raise InternalError(
                "Memory allocator node needs FakeTensor val or TensorMetadata to proceed"
            )
    elif isinstance(val, FakeTensor):
        alloc_spec = (val.shape, val.dtype)
    else:
        assert isinstance(val, list) or isinstance(val, tuple)
        assert isinstance(tensor_meta, list) or isinstance(tensor_meta, tuple)
        alloc_spec: List[memory.AllocSpec] = []
        for v, t in zip(val, tensor_meta):
            if v is not None:
                # pyre-fixme[6]: For 1st argument expected
                #  `Union[List[Tuple[List[int], dtype]], Tuple[List[int], dtype]]` but
                #  got `Tuple[Size, dtype]`.
                alloc_spec.append((v.shape, v.dtype))
            elif t is not None:
                # pyre-fixme[6]: For 1st argument expected
                #  `Union[List[Tuple[List[int], dtype]], Tuple[List[int], dtype]]` but
                #  got `Tuple[Size, dtype]`.
                alloc_spec.append((t.shape, t.dtype))
            else:
                raise InternalError(
                    "Memory allocator node needs FakeTensor val or TensorMetadata to proceed"
                )

    # pyre-fixme[6]
    alloc = graph_module.graph.call_function(memory.alloc, (alloc_spec,))
    alloc.meta["val"] = val
    alloc.meta["tensor_meta"] = tensor_meta
    return alloc


class ToOutVarPass(PassBase):
    def __init__(self, ignore_to_out_var_failure: bool = False) -> None:
        self.ignore_to_out_var_failure = ignore_to_out_var_failure

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        """
        Converts all of the functions to contain an out variant if it does not exist
        """
        missing_out_vars: Set[str] = set()

        def get_submodule(node: torch.fx.Node) -> torch.fx.GraphModule:
            assert node.op == "get_attr"
            return getattr(graph_module, node.target)

        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            target = node.target
            if target == torch.ops.higher_order.cond:
                self.call(get_submodule(node.args[1]))
                self.call(get_submodule(node.args[2]))
                continue
            if target == torch.ops.higher_order.map_impl:
                self.call(get_submodule(node.args[0]))
                continue
            elif target == control_flow.while_loop:
                self.call(get_submodule(node.args[0]))
                self.call(get_submodule(node.args[1]))
                continue
            elif getattr(target, "__module__", None) == "_operator":
                continue
            elif target in to_out_var_skiplist:
                continue
            if not isinstance(
                target, (torch._ops.OpOverload, EdgeOpOverload, BackendOpOverload)
            ):
                raise RuntimeError(f"Require an op overload for target: {target}")

            op_name = target._schema.name
            overload_name = target._schema.overload_name
            if is_out_variant(op_name, overload_name):
                # TODO (zhxchen17) Remove this after functionalization is always on.
                if "out" in node.kwargs and isinstance(node.kwargs["out"], fx.Node):
                    out = node.kwargs["out"]
                    if out.target is not memory.alloc and len(out.users) == 1:
                        with graph_module.graph.inserting_before(node):
                            alloc = make_alloc_node(
                                graph_module,
                                node.meta["val"],
                                node.meta["tensor_meta"],
                            )
                        out.replace_all_uses_with(alloc)
                        graph_module.graph.erase_node(out)
                continue

            try:
                if isinstance(target, (EdgeOpOverload, BackendOpOverload)):
                    out_var_target = target.to_out_variant()
                    out_args_names = get_out_args_from_opoverload(out_var_target)
                else:
                    out_var_target, out_args_names = to_out_variant(target)
            except RuntimeError as e:
                graph_module.encounter_to_out_var_failure = True
                logging.info(
                    f"Failed converting '{target}' to its out variant with error: '{e}'"
                )
                missing_out_vars.add(op_name)
                continue

            assert out_var_target
            out_var_kwargs = {}

            # Pool functional target's kwargs into out-variant's kwargs
            for arg in out_var_target._schema.arguments:
                if arg.name in out_args_names:
                    continue
                if arg.name in node.kwargs:
                    out_var_kwargs[arg.name] = node.kwargs[arg.name]

            with graph_module.graph.inserting_before(node):
                if len(out_args_names) == 1:
                    alloc_node = make_alloc_node(
                        graph_module, node.meta["val"], node.meta["tensor_meta"]
                    )
                    out_var_kwargs[out_args_names[0]] = alloc_node
                    if len(out_var_target._schema.returns) == 0:
                        node.replace_all_uses_with(alloc_node)
                else:
                    # If the op has multiple out args, we assume the node's
                    # metadata contains a fake tensor with the same size and type
                    fake_tensor_list = node.meta["val"]
                    tensor_metadatas = node.meta["tensor_meta"]
                    assert isinstance(
                        fake_tensor_list, (list, tuple)
                    ), "Expected a list/tuple of tensors when the op has multiple out arguments"
                    assert len(out_args_names) == len(
                        fake_tensor_list
                    ), f"Expected {len(out_args_names)} tensor specs, but got {len(node.meta['val'])}"
                    for out_arg_name, val, tensor_meta in zip(
                        out_args_names, fake_tensor_list, tensor_metadatas
                    ):
                        if val is None:
                            out_var_kwargs[out_arg_name] = None
                            continue
                        assert isinstance(val, FakeTensor)
                        out_var_kwargs[out_arg_name] = make_alloc_node(
                            graph_module, val, tensor_meta
                        )

            node.target = out_var_target
            node.kwargs = out_var_kwargs

        if (not self.ignore_to_out_var_failure) and len(missing_out_vars) > 0:
            raise RuntimeError(f"Missing out variants: {missing_out_vars}")
        return PassResult(graph_module, True)


def to_scratch_op_pass(graph_module: torch.fx.GraphModule) -> PassResult:
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if not isinstance(target, torch._ops.OpOverload):
            # ignore ops that are not OpOverload. Examples are operator.getitem,
            # memory.alloc etc.
            continue

        scratch_op = to_scratch_op(target)
        if not scratch_op:
            continue

        args_vals = [nd.meta.get("val") for nd in node.args]
        kwargs_vals = {name: nd.meta.get("val") for name, nd in node.kwargs.items()}
        get_scratch_metas = getattr(target, "get_scratch_metas", None)
        if not get_scratch_metas:
            raise RuntimeError(
                "The get_scratch_metas attribute is not found on the out variant op when converting it to a scratch op. Make sure you have imported the module that attaches the get_scratch_metas attribute to the out variant op."
            )
        scratch_metas = get_scratch_metas(*args_vals, **kwargs_vals)
        scratch_kwargs = {}
        with graph_module.graph.inserting_before(node):
            for name, val in scratch_metas.items():
                scratch = make_alloc_node(graph_module, val, None)
                scratch_kwargs[name] = scratch
        node.target = scratch_op
        kwargs = dict(node.kwargs)
        kwargs.update(scratch_kwargs)
        node.kwargs = kwargs
        logging.debug(f"Out variant {target} is converted to scratch op {scratch_op}")
    return PassResult(graph_module, True)


def dead_code_elimination_pass(graph_module: torch.fx.GraphModule) -> PassResult:
    for subgm in graph_module.modules():
        if not isinstance(subgm, torch.fx.GraphModule):
            continue
        subgm.graph.eliminate_dead_code()
        subgm.recompile()
    return PassResult(graph_module, True)


# Passes to convert a graph module from ATen to Edge IR

base_pre_op_replace_passes: List[Callable[[torch.nn.Module], PassResult]] = PassManager(
    passes=[
        # ReplaceSymSizeOpPass need to be run before other passes which inherits
        # from ExportPass. ExportPass can not handle OpOverloadPacket in its
        # call_function method. The ReplaceSymSizeOpPass pass converts sym size
        # ops from OpOverloadPacket to OpOverload.
        ReplaceSymSizeOpPass(),
        NormalizeTransposePass(),
        ReplaceBrokenOpsWithFunctionalOpsPass(),
        ScalarToTensorPass(),
        SymToTensorPass(),
        RemoveNoopPass(),
        RemoveToCopyPass(),
    ]
).passes

base_post_op_replace_passes: List[Callable[[torch.nn.Module], PassResult]] = (
    PassManager(
        passes=[
            dead_code_elimination_pass,
            DebugHandleGeneratorPass(),
        ]
    ).passes
)


def propagate_dynamic_shape(
    dynamic_memory_planning_mode: DynamicMemoryPlanningMode = DynamicMemoryPlanningMode.UPPER_BOUND,
) -> List[PassType]:
    """
    Run a few passes on the GraphModule to propagate the dynamic shape information.

    Mainly used to provide dynamic shape information for delegation.
    """
    return [
        SpecPropPass(),
        HintBasedSymShapeEvalPass(),
    ]
