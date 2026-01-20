# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
import os
import re
import warnings
from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor

import executorch.exir as exir
import torch

from executorch.backends.qualcomm._passes import AnnotateStack, AnnotateUnbind
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager

from executorch.backends.qualcomm.builders.node_visitor import (
    QNN_QUANT_TYPE_MAP,
    QNN_TENSOR_TYPE_MAP,
)
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.partition.qnn_partitioner import (
    generate_qnn_executorch_option,
    get_skip_decomp_table,
    QnnPartitioner,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    HtpArch,
    QcomChipset,
    QnnExecuTorchBackendOptions,
    QnnExecuTorchBackendType,
    QnnExecuTorchGpuBackendOptions,
    QnnExecuTorchGpuPrecision,
    QnnExecuTorchHtpBackendOptions,
    QnnExecuTorchHtpPerformanceMode,
    QnnExecuTorchHtpPrecision,
    QnnExecuTorchLogLevel,
    QnnExecuTorchOpPackageOptions,
    QnnExecuTorchOptions,
    QnnExecuTorchProfileLevel,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
    option_to_flatbuffer,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_QNN_COMPILE_SPEC,
    QCOM_QUANTIZED_IO,
)
from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import QnnManagerContext

from executorch.exir import EdgeCompileConfig, ExirExportedProgram, to_edge
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.program._program import (
    EdgeProgramManager,
    to_edge_transform_and_lower,
)
from tabulate import tabulate
from torch._decomp import core_aten_decompositions, remove_decompositions
from torch.export.exported_program import ExportedProgram
from torch.fx import passes
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.library import Library


class _AnnotationSkipper(OperatorSupportBase):
    """
    Class used to partition out unwanted graph nodes.
    e.g. - nodes are prevented from quantization annotation
         - nodes have been grouped together as a submodule

    Attributes
    ----------
    fp_node_id_set : set
        a set contains nodes' name to be left in fp precision
    fp_node_op_set : set
        a set contains nodes' target (aten dialect) to be left in fp precision
    skip_annotated_submodule : bool
        flag to skip annotated submodule or not

    Methods
    -------
    should_delegate(n: torch.fx.Node)
        identify the residual nodes haven't be lowered with fixed-precision
    should_skip(n: torch.fx.Node)
        identify the nodes should be kept out with fixed-precision or not
    is_node_supported(_, node: torch.fx.Node)
        overridden method for graph partitioning
    """

    def __init__(
        self,
        fp_node_id_set: set = None,
        fp_node_op_set: set = None,
        skip_annotated_submodule: bool = False,
    ):
        self.fp_node_id_set = fp_node_id_set
        self.fp_node_op_set = fp_node_op_set
        self.skip_annotated_submodule = skip_annotated_submodule

    def should_delegate(self, n: torch.fx.Node):
        return n.op == "call_function" and n.target != operator.getitem

    def should_skip(self, n: torch.fx.Node):
        return n.name in self.fp_node_id_set or n.target in self.fp_node_op_set

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        if self.skip_annotated_submodule:
            if node.op == "get_attr":
                return all(self.should_delegate(user) for user in node.users)
            return self.should_delegate(node)

        if any(
            [
                node.op in ("placeholder", "output"),
                self.should_skip(node),
                # check if parameters belong to fallbacked operator
                (
                    node.op == "get_attr"
                    and all(self.should_skip(user) for user in node.users)
                ),
            ]
        ):
            print(f"[QNN Quantizer Annotation]: {node.name} | Skipped")
            return False

        return True


def qnn_capture_config():
    return exir.CaptureConfig(enable_aot=True)


def qnn_edge_config() -> exir.EdgeCompileConfig:
    return exir.EdgeCompileConfig(
        _check_ir_validity=False,
    )


def convert_linear_to_conv2d(module: torch.nn.Module):
    class Conv2D(torch.nn.Module):
        def __init__(self, weight, bias=None):
            super().__init__()
            use_bias = bias is not None
            self.conv = torch.nn.Conv2d(
                in_channels=weight.shape[0],
                out_channels=weight.shape[1],
                kernel_size=1,
                padding=0,
                bias=use_bias,
            )
            self.conv.weight = torch.nn.Parameter(weight.reshape(*weight.shape, 1, 1))
            if use_bias:
                self.conv.bias = torch.nn.Parameter(bias)

        def forward(self, x):
            rank = x.dim()
            x = x.reshape(*x.shape, 1) if rank == 3 else x.reshape(1, *x.shape, 1)
            x = torch.transpose(x, 1, 2)
            res = self.conv(x)
            res = torch.transpose(res, 1, 2)
            res = res.squeeze(-1) if rank == 3 else res.reshape(*res.shape[1:3])
            return res

    def replace_linear(module: torch.nn.Module):
        attr_strs = dir(module)
        if isinstance(module, torch.nn.ModuleList):
            attr_strs += [str(i) for i in range(len(module))]

        for attr_str in attr_strs:
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, torch.nn.Linear):
                setattr(module, attr_str, Conv2D(target_attr.weight, target_attr.bias))

        for _, sub_module in module.named_children():
            sub_module = replace_linear(sub_module)
        return module

    return replace_linear(module)


def dump_context_from_pte(pte_path) -> List[str]:
    """
    Dump compiled binaries under the same directory of pte_path.
    For partitioned graph, there will be multiple files with names f"{method_name}_{index}".
    'method_name' refers to the name of a method in the nn.Module that was traced to
    generate this program, while 'index' indicates the order of execution.

    Args:
        pte_path (str): The path of generated pte.
    """
    import os

    from executorch.exir._serialize._program import deserialize_pte_binary

    with open(pte_path, "rb") as f:
        program_data = f.read()

    program = deserialize_pte_binary(program_data).program

    ctx_path = os.path.dirname(pte_path)
    dumpfiles = []
    for execution_plan in program.execution_plan:
        for i, delegate in enumerate(execution_plan.delegates):
            if delegate.id == "QnnBackend":
                processed_bytes = program.backend_delegate_data[
                    delegate.processed.index
                ].data
                binary = PyQnnManagerAdaptor.StripProtocol(processed_bytes)
                file_extension = ".bin"
                if len(binary) == 0:
                    binary = processed_bytes
                    file_extension = ".dlc"
                dump_file = f"{ctx_path}/{execution_plan.name}_{i}{file_extension}"
                with open(dump_file, "wb") as f:
                    f.write(binary)
                dumpfiles.append(dump_file)
    return dumpfiles


def update_spill_fill_size(
    exported_program: ExportedProgram | List[LoweredBackendModule],
):
    # check if user specifies to use multi_contexts
    # this is a generic approach in case there exists multiple backends
    def get_program_info(program):
        def process_exported_program(prog):
            max_sf_buf_size, module_map = 0, {}
            for _, m in prog.graph_module._modules.items():
                # currently only 1 compile spec is expected in each partition
                options = flatbuffer_to_option(m.compile_specs[0].value)
                if (
                    options.backend_options.backend_type
                    == QnnExecuTorchBackendType.kHtpBackend
                    and options.backend_options.htp_options.use_multi_contexts
                ):
                    qnn_mgr = PyQnnManagerAdaptor.QnnManager(
                        m.compile_specs[0].value, m.processed_bytes
                    )
                    assert qnn_mgr.Init().value == 0, "failed to load context binary"
                    max_sf_buf_size = max(
                        max_sf_buf_size, qnn_mgr.GetSpillFillBufferSize()
                    )
                    module_map[m] = options
                    qnn_mgr.Destroy()
            return max_sf_buf_size, module_map

        def process_lowered_module(module):
            qnn_mgr = PyQnnManagerAdaptor.QnnManager(
                module.compile_specs[0].value, module.processed_bytes
            )
            assert qnn_mgr.Init().value == 0, "failed to load context binary"
            spill_fill_size = qnn_mgr.GetSpillFillBufferSize()
            qnn_mgr.Destroy()
            return spill_fill_size, {
                module: flatbuffer_to_option(module.compile_specs[0].value)
            }

        dispatch = {
            ExportedProgram: process_exported_program,
            LoweredBackendModule: process_lowered_module,
        }
        return dispatch[type(program)](program)

    def update_program(max_sf_buf_size, module_map):
        def set_spec(module, options):
            spec = CompileSpec(QCOM_QNN_COMPILE_SPEC, option_to_flatbuffer(options))
            if isinstance(module, ExportedProgram):
                module.compile_specs[0] = spec
            else:
                module._compile_specs[0] = spec

        for module, options in module_map.items():
            options.backend_options.htp_options.max_sf_buf_size = max_sf_buf_size
            set_spec(module, options)

    max_sf_size, modules_map = 0, {}
    if isinstance(exported_program, list):
        for prog in exported_program:
            max_sf_buf_size, module_map = get_program_info(prog)
            max_sf_size = max(max_sf_size, max_sf_buf_size)
            modules_map.update(module_map)
    else:
        max_sf_size, module_map = get_program_info(exported_program)
    update_program(max_sf_size, module_map)

    return max_sf_size


def canonicalize_program(obj):
    update_spill_fill_size(obj)


def get_decomp_table(passes_job) -> Dict[torch._ops.OperatorBase, Callable]:
    source_decompositions = core_aten_decompositions()
    # The below super ops are supported by QNN
    skip_decompositions = get_skip_decomp_table()

    # If we want to annotate the decomposed ops, then we should decompose the operation.
    if passes_job:
        skip_decompositions = [
            skip_decomp_op
            for skip_decomp_op in skip_decompositions
            if skip_decomp_op
            not in AnnotateStack.decomp_ops + AnnotateUnbind.decomp_ops
        ]
    remove_decompositions(source_decompositions, skip_decompositions)

    return source_decompositions


def to_edge_transform_and_lower_to_qnn(
    module: Union[
        torch.nn.Module,
        torch.fx.GraphModule,
        Dict[str, torch.nn.Module],
        Dict[str, torch.fx.GraphModule],
    ],
    inputs: Union[Tuple[torch.Tensor], Dict[str, Tuple[torch.Tensor]]],
    compiler_specs: Union[List[Any], Dict[str, List[Any]]],
    constant_methods: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Dict] = None,
    dep_table: Optional[Dict] = None,
    passes_job: Optional[Union[OrderedDict, Dict[str, OrderedDict]]] = None,
    skip_node_id_set: Optional[set] = None,
    skip_node_op_set: Optional[set] = None,
    skip_mutable_buffer: Optional[bool] = False,
    generate_etrecord: Optional[bool] = False,
    convert_linear_to_conv2d: Optional[bool] = False,
) -> EdgeProgramManager:
    """
    Transforms and lowers a given PyTorch module to the QNN backend.

    Args:
        module (Union[torch.nn.Module, torch.fx.GraphModule,Dict[str, torch.nn.Module], Dict[str, torch.fx.GraphModule]]):
            The PyTorch module or fx.GraphModule to be transformed.
        inputs (Union[Tuple[torch.Tensor], Dict[str, Tuple[torch.Tensor]]]):
            The input tensors for the module.
        compiler_specs (Union[List[Any], Dict[str, List[Any]]]):
            Compiler specifications for Qualcomm AI Engine Direct.
        constant_methods (Optional[Dict[str, Any]]):
            An optional dictionary mapping method names to constant values returned by those methods in eager mode.
            Often used to store configuration information on Edge models.
        dynamic_shapes (Optional[Dict]):
            Information about dynamic shapes.
        dep_table (Optional[Dict]):
            Dependency table for the transformation passes.
        passes_job (Optional[Union[OrderedDict, Dict[str, OrderedDict]]]):
            Ordered dictionary of transformation passes.
        skip_node_id_set (Optional[set]):
            Set of node IDs to skip during partitioning.
        skip_node_op_set (Optional[set]):
            Set of node operations to skip during partitioning.
        skip_mutable_buffer (Optional[bool]):
            Whether to skip delegating the mutable buffer in QNN backend.
        convert_linear_to_conv2d (Optional[bool]):
            Whether to convert linear to conv2d in some cases to improve performance in HTP backend.

    Returns:
        EdgeProgramManager:
            The manager for the edge program after transformation and lowering.
    """

    def ensure_graph_specific_dict(value, graph_names):
        """
        Ensures the input value is a dictionary with keys matching the provided graph names.
        If the input is not a dictionary or its keys do not match the graph names, a new dictionary
        is created with the graph names as keys and the input value assigned to each key.

        Examples:
            1. Input is None:
                >>> ensure_graph_specific_dict(None, ["forward1", "forward2"])
                {'forward1': None, 'forward2': None}

            2. Input is a single value:
                >>> ensure_graph_specific_dict(input, ["forward1", "forward2"])
                {'forward1': input, 'forward2': input}

            3. Input is a non-graph specific dict:
                >>> ensure_graph_specific_dict({Any: input}, ["forward1", "forward2"])
                {'forward1': {Any: input}, 'forward2': {Any: input}}
        """
        if value is None:
            return {graph_name: None for graph_name in graph_names}
        if isinstance(value, dict) and graph_names == value.keys():
            return value
        return {graph_name: value for graph_name in graph_names}

    # Ensure if user is using intermediate debugger, user only lower 1 method.
    # This restriction is caused by conflict handle_id among graphs.
    # This could be resolved with generating random debug_id(e.g., uuid).
    for compiler_spec in (
        compiler_specs.values()
        if isinstance(compiler_specs, Dict)
        else [compiler_specs]
    ):
        option = generate_qnn_executorch_option(compiler_spec)
        obj_options = flatbuffer_to_option(option)
        if obj_options.dump_intermediate_outputs and isinstance(module, Dict):
            assert (
                len(module) == 1
            ), "Intermediate Tensor Dump does not support multi-methods."

    if not isinstance(module, dict):
        module = {"forward": module}

    # Ensure attributes are graph-specific dictionaries
    graph_names = module.keys()
    inputs = ensure_graph_specific_dict(inputs, graph_names)
    compiler_specs = ensure_graph_specific_dict(compiler_specs, graph_names)
    dynamic_shapes = ensure_graph_specific_dict(dynamic_shapes, graph_names)
    dep_table = ensure_graph_specific_dict(dep_table, graph_names)
    passes_job = ensure_graph_specific_dict(passes_job, graph_names)

    # Prepare programs and partitioners
    aten_programs = {}
    transform_passes = {}
    qnn_partitioners = {
        graph_name: [
            QnnPartitioner(
                compiler_specs[graph_name],
                skip_node_id_set=skip_node_id_set,
                skip_node_op_set=skip_node_op_set,
                skip_mutable_buffer=skip_mutable_buffer,
            )
        ]
        for graph_name in graph_names
    }

    for graph_name, m in module.items():
        ep = torch.export.export(
            m,
            inputs[graph_name],
            dynamic_shapes=dynamic_shapes[graph_name],
            strict=True,
        )
        # This transformation is primarily intended for the LiftConstantScalarOperands pass
        # to avoid creating temporary tensors in the operation builder.
        # However, this pass will create a get_attr node, which should be converted
        # into a lifted tensor constant by the lift_constant_tensor_pass.
        # If placed in the to_edge_transform_passes, it will be executed
        # after the lift_constant_tensor_pass, causing the operation builder
        # to fail to correctly retrieve the parameter by the get_parameter.
        aten_programs[graph_name] = QnnPassManager().transform_for_export_pipeline(
            ep, convert_linear_to_conv2d=convert_linear_to_conv2d
        )
        transform_passes[graph_name] = QnnPassManager().get_to_edge_transform_passes(
            ep, passes_job=passes_job[graph_name], dep_table=dep_table[graph_name]
        )
    with QnnManagerContext(compiler_specs):
        return to_edge_transform_and_lower(
            aten_programs,
            transform_passes=transform_passes,
            partitioner=qnn_partitioners,
            constant_methods=constant_methods,
            compile_config=qnn_edge_config(),
            generate_etrecord=generate_etrecord,
        )


def capture_program(
    module: Union[torch.nn.Module, torch.fx.GraphModule],
    inputs: Tuple[torch.Tensor],
    dep_table: Optional[Dict] = None,
    passes_job: OrderedDict = None,
    dynamic_shapes: Dict = None,
) -> exir.ExirExportedProgram:
    """
    TODO: Deprecated capture_program with to_edge_transform_and_lower_to_qnn

    Captures and transforms a PyTorch module into an Exir exported program.

    Args:
        module (Union[torch.nn.Module, torch.fx.GraphModule]): The PyTorch module or fx.GraphModule to be captured.
        inputs (Tuple[torch.Tensor]): The input tensors for the module.
        dep_table (Optional[Dict]): Dependency table for the transformation passes.
        passes_job (OrderedDict, optional): Ordered dictionary of transformation passes.
        dynamic_shapes (Dict, optional): Information about dynamic shapes.

    Returns:
        exir.ExirExportedProgram: The transformed Exir exported program ready for lowering to QNN backend.
    """
    warnings.warn(
        "capture_program is deprecated. Use to_edge_transform_and_lower_to_qnn instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    ep = torch.export.export(module, inputs, dynamic_shapes=dynamic_shapes, strict=True)
    ep = QnnPassManager().transform_for_export_pipeline(ep)
    # TODO: Handle stack op. If we want to run annotate_decomposed pass for stack op,
    # we need to make stack op decompose, which means we need to find a method to
    # remove it from skip_decomp table
    decomposed_ep = ep.run_decompositions(get_decomp_table(passes_job))
    core_ep = ExirExportedProgram(decomposed_ep, False)
    edge_ep = core_ep.to_edge(qnn_edge_config())
    transform_passes = QnnPassManager().get_to_edge_transform_passes(
        edge_ep.exported_program,
        passes_job=passes_job,
        dep_table=dep_table,
    )
    edge_ep.transform(*transform_passes)
    return edge_ep


def _partition_graph_into_submodules(gm, subgm_tag, subgm_cb, ptn):
    from torch.fx.passes.utils.fuser_utils import (
        erase_nodes,
        fuse_as_graphmodule,
        insert_subgm,
        legalize_graph,
        topo_sort,
    )

    partitions = ptn.propose_partitions()
    # insert meta for each partition group
    for i, partition in enumerate(partitions):
        for node in partition.nodes:
            node.meta[subgm_tag] = i

    for i in range(len(partitions)):
        # find nodes with same group id in current graph
        node_list = [
            node for node in gm.graph.nodes if node.meta.get(subgm_tag, "") == i
        ]
        # fuse group nodes into submodule
        sorted_nodes = topo_sort(node_list)
        submodule_name = f"{subgm_tag}_{i}"
        subgm, orig_inputs, orig_outputs = fuse_as_graphmodule(
            gm, sorted_nodes, submodule_name
        )
        # insert submodule & trim group nodes
        gm = insert_subgm(
            gm,
            subgm_cb(subgm, submodule_name),
            orig_inputs,
            orig_outputs,
        )
        erase_nodes(gm, sorted_nodes)
        legalize_graph(gm)

    gm.recompile()
    return gm


def _canonicalize_graph_with_lowered_module(gm, subgm_tag, compiler_specs):
    # return lowered program for user to debug
    edge_prog_mgrs = []
    # partition each submodule which went through convert_pt2e
    for node in gm.graph.nodes:
        if node.op == "call_module" and subgm_tag in node.name:
            # obtain sample inputs through meta
            subgm_input = [
                torch.ones(arg.meta["val"].shape, dtype=arg.meta["val"].dtype)
                for arg in node.args
            ]
            # start lowering with given partitioner
            edge_prog_mgrs.append(
                to_edge_transform_and_lower_to_qnn(
                    gm.get_submodule(node.name), tuple(subgm_input), compiler_specs
                )
            )
            # replace submodule with lowered module
            gm.set_submodule(
                node.name,
                edge_prog_mgrs[-1].exported_program().graph_module,
            )
            # if node has multiple outputs, getitems will be default generated
            if all(n.target != operator.getitem for n in node.users):
                with gm.graph.inserting_after(node):
                    getitem_node = gm.graph.call_function(
                        operator.getitem,
                        (node, 0),
                    )
                    getitem_node.meta = node.meta
                    node.replace_all_uses_with(
                        replace_with=getitem_node,
                        delete_user_cb=lambda user: user.target != operator.getitem,
                    )

    gm.recompile()
    return gm, edge_prog_mgrs


def skip_annotation(
    nn_module: torch.nn.Module,
    quantizer,
    compiler_specs,
    sample_input: Tuple[torch.Tensor, ...],
    calibration_cb: Callable[[torch.fx.GraphModule], None],
    fp_node_id_set: set = None,
    fp_node_op_set: set = None,
    fallback_to_cpu: bool = True,
):
    r"""
    Exclude speific operators from quantizer annotation.
    Skipped operators will defaultly stay in CPU, set 'fallback_to_cpu'
    to False for trying to delegate them with FP16 precision.

    e.g.: consider following graph:
    bias_1 weight_1 input_1   bias_2 weight_2 input_2
      | (placeholder) |         | (placeholder) |
       \      |      /           \      |      /
        \     |     /             \     |     /
         \    |    /               \    |    /
           conv2d_1                 conv2d_2
           (torch.ops.aten.conv2d.default)
               \                       /
                \                     /
                 \_______     _______/
                         add_1
             (torch.ops.aten.add.default)
                           |
                         output

    If user wants to skip convolution op by names with
    'skip_node_id_set' = {"conv2d_1"}
    "bias_1 / weight_1 / input_1 / input_2 / conv2d_1"
    will be partitioned out and not annotated / lowered with QNN.

    [Generated graph]
    bias_1 weight_1 input_1   input_2
      | (placeholder) |          |
       \      |      /           |
        \     |     /            |
         \    |    /             |
           conv2d_1              |
              \                 /
               \               /
                \             /
               lowered_module_1
            (QNN fixed precision)
                      |
                    output

    If user wants to skip convolution op by target with
    'skip_node_op_set' = {torch.ops.aten.conv2d.default}
    "bias_1 / weight_1 / input_1 / conv2d_1,
     bias_2 / weight_2 / input_2 / conv2d_2"
    will be partitioned out and not annotated / lowered with QNN.

    [Generated graph]
    bias_1 weight_1 input_1   bias_2 weight_2 input_2
      | (placeholder) |         | (placeholder) |
       \      |      /           \      |      /
        \     |     /             \     |     /
         \    |    /               \    |    /
           conv2d_1                 conv2d_2
           (torch.ops.aten.conv2d.default)
               \                       /
                \                     /
                 \__               __/
                    lowered_module_1
                 (QNN fixed precision)
                           |
                         output

    If user wants to delegate the skipped conv2d from above graph
    with 'fallback_to_cpu' = False:

    [Generated graph]
       input_1         input_2
    (placeholder)   (placeholder)
          |               |
          \               /
          lowered_module_2
         (QNN fp16 precision)
                  |
                  |
          lowered_module_1
         (QNN fixed precision)
                  |
                output

    Args:
        nn_module (torch.nn.Module): The module to be lowered.
        quantizer (QnnQuantizer): Instance of QnnQuantizer.
        compiler_specs (List[CompileSpec]): Compiler specs for Qualcomm AI Engine Direct.
        sample_input ((torch.Tensor, ...)): Sample input tensors for graph exporting.
        calibration_cb (callable): Callback function for user-defined calibration.
        fp_node_id_set ({str, ...}): Set of operator names to be left in fp precision.
        fp_node_op_set ({torch.ops.aten.xxx, ...}): Set of operator targets to be left in fp precision.
        fallback_to_cpu (bool): Whether to lower skipped nodes to fp16 or not.

    Returns:
        exported_programs: List of programs lowered to QnnBackend (quantized graphs only).
    """
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchHtpPrecision,
    )
    from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
        flatbuffer_to_option,
    )
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    def prepare_subgm(subgm, subgm_name):
        # prepare current submodule for quantization annotation
        subgm_prepared = prepare_pt2e(subgm, quantizer)
        # overwrite this attribute or name will be set to "GraphModule"
        # we could not identify each submodule if action is not performed
        subgm_prepared.__class__.__name__ = subgm_name
        return subgm_prepared

    fp_node_id_set = fp_node_id_set if fp_node_id_set is not None else set()
    fp_node_op_set = fp_node_op_set if fp_node_op_set is not None else set()
    graph_module = torch.export.export(nn_module, sample_input, strict=True).module()
    # define node support type
    capability_partitioner = CapabilityBasedPartitioner(
        graph_module,
        _AnnotationSkipper(fp_node_id_set, fp_node_op_set),
        allows_single_node_partition=True,
    )
    subgm_tag = "annotated_group"
    graph_module = _partition_graph_into_submodules(
        gm=graph_module,
        subgm_tag=subgm_tag,
        subgm_cb=prepare_subgm,
        ptn=capability_partitioner,
    )
    # perform calibration
    calibration_cb(graph_module)
    # convert sub modules which went through prepare_pt2e
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            graph_module.set_submodule(
                node.name,
                convert_pt2e(graph_module.get_submodule(node.name)),
            )
    # canonicalize graph for lowering again
    graph_module, edge_prog_mgrs = _canonicalize_graph_with_lowered_module(
        gm=graph_module,
        subgm_tag=subgm_tag,
        compiler_specs=compiler_specs,
    )

    if not fallback_to_cpu:
        try:
            # change HTP compiler spec for hardware to enable fp16
            qnn_option = generate_qnn_executorch_option(compiler_specs)
            compile_option = flatbuffer_to_option(qnn_option)
            htp_options = compile_option.backend_options.htp_options
            htp_options.precision = QnnExecuTorchHtpPrecision.kHtpFp16
            compiler_specs[0].value = option_to_flatbuffer(compile_option)
        except:
            print(
                "Failed to change HTP compiler spec with 'use_fp16' as True,"
                " skipped operators will fallback to cpu,"
            )
            return graph_module, edge_prog_mgrs

        # try lowering skipped operator into fp16
        capability_partitioner = CapabilityBasedPartitioner(
            graph_module,
            _AnnotationSkipper(skip_annotated_submodule=True),
            allows_single_node_partition=True,
        )
        subgm_tag = "skipped_group"
        graph_module = _partition_graph_into_submodules(
            gm=graph_module,
            subgm_tag=subgm_tag,
            subgm_cb=lambda subgm, _: subgm,
            ptn=capability_partitioner,
        )
        graph_module, edge_prog_mgrs_fp = _canonicalize_graph_with_lowered_module(
            gm=graph_module,
            subgm_tag=subgm_tag,
            compiler_specs=compiler_specs,
        )
        edge_prog_mgrs.extend(edge_prog_mgrs_fp)

    return graph_module, edge_prog_mgrs


def from_context_binary(  # noqa: C901
    ctx_path: str | bytes,
    op_name: str,
    soc_model: QcomChipset = QcomChipset.SM8650,
    custom_info: Dict = None,
):
    from pathlib import Path

    def implement_op(custom_op, op_name, outputs):
        @torch.library.impl(
            custom_op, str(op_name), dispatch_key="CompositeExplicitAutograd"
        )
        def op_impl(inputs: List[torch.Tensor]):
            return tuple(
                torch.zeros(tuple(v.shape), device="meta", dtype=v.dtype)
                for v in outputs.values()
            )

    def build_graph(
        inputs,
        outputs,
        qnn_in_order: Optional[List[int]] = None,
        executorch_in_order: Optional[List[int]] = None,
        executorch_out_order: Optional[List[int]] = None,
    ):
        # custom op declaration
        inputs_str = "Tensor[] inputs"
        func_proto = f"{op_name}({inputs_str}) -> Any"
        custom_op = Library(OpContextLoader.namespace, "FRAGMENT")
        custom_op.define(func_proto)
        # custom op implementation
        implement_op(custom_op, op_name, outputs)

        # model architecture mimicking context binary
        class Model(torch.nn.Module):
            """
            The args of forward() can be thought of as what executorch is accepting as input.
            The getattr inside the forward() can be thought of as qnn context binary.
            When we first pass in the input, we need to use the executorch's(nn.module) input order.
            After we get into forward(), we then need to convert input order to qnn's input order.
            Same as return, when qnn returns the value, we need to reorder them back to executorh's output order.
            """

            def __init__(self, qnn_in_order, executorch_out_order):
                super().__init__()
                self.qnn_in_order = qnn_in_order
                self.executorch_out_order = executorch_out_order

            def forward(self, *inputs):  # executorch
                if self.qnn_in_order:
                    inputs = tuple(inputs[i] for i in self.qnn_in_order)
                ret = getattr(
                    getattr(torch.ops, OpContextLoader.namespace), op_name
                ).default(inputs)
                return (
                    [ret[idx] for idx in self.executorch_out_order]
                    if self.executorch_out_order
                    else ret
                )

        inputs = (
            tuple(tuple(inputs.values())[i] for i in executorch_in_order)
            if executorch_in_order
            else tuple(inputs.values())
        )

        model = Model(qnn_in_order, executorch_out_order)
        prog = torch.export.export(model, inputs, strict=True)
        # bookkeeping for variables' life cycle
        return {
            "custom_op": custom_op,
            "custom_module": model,
            "exported_program": prog,
        }

    def build_tensor(tensors, dtype_map):
        ret = OrderedDict()
        for t in tensors:
            dtype = t.GetDataType()
            dtype_torch = dtype_map.get(dtype, None)
            assert dtype_torch is not None, f"unknown qnn data type {dtype}"
            ret[t.GetName()] = torch.zeros(tuple(t.GetDims()), dtype=dtype_torch)

        return ret

    def preprocess_binary(ctx_bin, compiler_specs):
        qnn_mgr = PyQnnManagerAdaptor.QnnManager(
            generate_qnn_executorch_option(compiler_specs),
        )
        return bytes(qnn_mgr.MakeBinaryInfo(ctx_bin))

    # dummy compiler spec would be fine, since we're not compiling
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_model,
        backend_options=backend_options,
        is_from_context_binary=True,
    )

    ctx_bin = (
        ctx_path
        if not isinstance(ctx_path, str)
        else preprocess_binary(Path(f"{ctx_path}").read_bytes(), compiler_specs)
    )

    dtype_map = {}
    for type_map in (QNN_QUANT_TYPE_MAP, QNN_TENSOR_TYPE_MAP):
        for k, v in type_map.items():
            dtype_map.setdefault(v, k)

    qnn_in_order, executorch_in_order, executorch_out_order = None, None, None
    if custom_info is not None:
        # since some context binaries might fail to open on host
        # if they are compiled with special flags:
        # e.g. weight sharing
        # use custom information here instead
        inputs = build_tensor(custom_info["graph_inputs"], dtype_map)
        outputs = build_tensor(custom_info["graph_outputs"], dtype_map)
        qnn_in_order = custom_info.get("qnn_in_order", None)
        executorch_in_order = custom_info.get("executorch_in_order", None)
        executorch_out_order = custom_info.get("executorch_out_order", None)
        graph_name = custom_info["graph_name"]
    else:
        # get context-binary io tensor info through qnn manager
        qnn_mgr = PyQnnManagerAdaptor.QnnManager(
            generate_qnn_executorch_option(compiler_specs),
            ctx_bin,
        )
        assert qnn_mgr.Init().value == 0, "failed to load context binary"
        # assume we only have one graph in current context
        graph_name = qnn_mgr.GetGraphNames()[0]
        qnn_mgr.AllocateTensor(graph_name)
        inputs = build_tensor(qnn_mgr.GetGraphInputs(graph_name), dtype_map)
        outputs = build_tensor(qnn_mgr.GetGraphOutputs(graph_name), dtype_map)
        qnn_mgr.Destroy()
    # generate graph specific for loading context
    bundle_prog = build_graph(
        inputs, outputs, qnn_in_order, executorch_in_order, executorch_out_order
    )
    bundle_prog.update({"inputs": inputs, "outputs": outputs})

    # TODO: to_edge() decorator alters the function call behavior, which
    # requires "self" when calling. To work around this issue,
    # temporarily remove the first parameter name.
    edge_prog_mgr = to_edge(
        {graph_name: bundle_prog["exported_program"]},
        # do not alter name for custom op
        compile_config=EdgeCompileConfig(_use_edge_ops=False),
    )

    # update meta with context binary
    for n in edge_prog_mgr._edge_programs[graph_name].graph.nodes:
        if n.op == "call_function" and OpContextLoader.namespace in str(n.target):
            n.meta[OpContextLoader.meta_ctx_bin] = ctx_bin
            break

    bundle_prog["edge_program_manager"] = edge_prog_mgr.to_backend(
        QnnPartitioner(compiler_specs)
    )
    return bundle_prog


def draw_graph(title, path, graph_module: torch.fx.GraphModule):
    graph = passes.graph_drawer.FxGraphDrawer(graph_module, title)
    with open(f"{path}/{title}.svg", "wb") as f:
        f.write(graph.get_dot_graph().create_svg())


def generate_gpu_compiler_spec(
    precision: QnnExecuTorchGpuPrecision = QnnExecuTorchGpuPrecision.kGpuPrecisionUserProvided,
    use_memory_optimizations: bool = True,
    use_node_optimizations: bool = True,
    use_queue_recording: bool = True,
    use_weight_sharing: bool = False,
) -> QnnExecuTorchBackendOptions:
    """
    Helper function generating backend options for QNN HTP

    Args:
        precision:
            kGpuPrecisionFp32 - Sets the precision mode to floating point 32-bit (FP32).
            kGpuPrecisionFp16 - Sets the precision mode to floating point 16-bit (FP16).
            kGpuPrecisionHybrid - Sets the precision mode to FP16 for storage and FP32 for calculations.
            kGpuPrecisionUserProvided - Uses the tensor data type provided by the user.
        use_memory_optimizations: If true, backend will share NATIVE tensor memory
            based upon analysis of the network topology.
        use_node_optimizations: If true, backend will fuse compatible operations into
            one operation to improve performance.
        use_queue_recording: If true, backend will use queue recording to improve performance.
        use_weight_sharing: Used with multiple_graphs, where model size will be
            reduced when operations have the same weights across multiple graphs.

    Returns:
        QnnExecuTorchGpuBackendOptions: backend options for QNN GPU.
    """
    # TODO: enable performance hint mechanism in runtime and make this as an option
    gpu_options = QnnExecuTorchGpuBackendOptions()
    gpu_options.precision = precision
    gpu_options.use_memory_optimizations = use_memory_optimizations
    gpu_options.use_node_optimizations = use_node_optimizations
    gpu_options.use_queue_recording = use_queue_recording
    gpu_options.use_weight_sharing = use_weight_sharing

    return QnnExecuTorchBackendOptions(
        backend_type=QnnExecuTorchBackendType.kGpuBackend,
        gpu_options=gpu_options,
    )


def generate_htp_compiler_spec(
    use_fp16: bool,
    use_dlbc: bool = False,
    use_multi_contexts: bool = False,
    use_weight_sharing: bool = False,
) -> QnnExecuTorchBackendOptions:
    """
    Helper function generating backend options for QNN HTP

    Args:
        use_fp16: If true, the model is compiled to QNN HTP fp16 runtime.
            Note that not all SoC support QNN HTP fp16. Only premium tier SoC
            like Snapdragon 8 Gen 1 or newer can support HTP fp16.
        use_dlbc: Deep Learning Bandwidth Compression allows inputs to be
            compressed, such that the processing bandwidth can be lowered.
        use_multi_contexts: When multiple contexts are generated inside the same
            pte, it is possible to reserve a single spill-fill allocation that
            could be re-used across all the splits.
        use_weight_sharing: Used with multiple_graphs, where model size will be
            reduced when operations have the same weights across multiple graphs.

    Returns:
        QnnExecuTorchHtpBackendOptions: backend options for QNN HTP.
    """
    htp_options = QnnExecuTorchHtpBackendOptions()
    htp_options.precision = (
        QnnExecuTorchHtpPrecision.kHtpFp16
        if use_fp16
        else QnnExecuTorchHtpPrecision.kHtpQuantized
    )
    # This actually is not an option which can affect the compiled blob.
    # But we don't have other place to pass this option at execution stage.
    # TODO: enable voting mechanism in runtime and make this as an option
    htp_options.performance_mode = QnnExecuTorchHtpPerformanceMode.kHtpBurst
    htp_options.use_multi_contexts = use_multi_contexts
    htp_options.use_weight_sharing = use_weight_sharing
    htp_options.use_dlbc = use_dlbc
    return QnnExecuTorchBackendOptions(
        backend_type=QnnExecuTorchBackendType.kHtpBackend,
        htp_options=htp_options,
    )


def generate_qnn_executorch_compiler_spec(
    soc_model: QcomChipset,
    backend_options: QnnExecuTorchBackendOptions,
    debug: bool = False,
    saver: bool = False,
    online_prepare: bool = False,
    dump_intermediate_outputs: bool = False,
    profile: bool = False,
    optrace: bool = False,
    shared_buffer: bool = False,
    is_from_context_binary: bool = False,
    op_package_options: QnnExecuTorchOpPackageOptions = None,
    use_mha2sha: bool = False,
) -> List[CompileSpec]:
    """
    Helper function generating compiler specs for Qualcomm AI Engine Direct

    Args:
        soc_model: The SoC you plan to run the compiled model. Please check
            QcomChipset for supported SoC.
            SM8450 (Snapdragon 8 Gen 1)
            SM8475(Snapdragon 8 Gen 1+)
            SM8550(Snapdragon 8 Gen 2)
            SM8650(Snapdragon 8 Gen 3)
            SM8750(Snapdragon 8 Elite)
            SM8850(Snapdragon 8 Elite Gen 5)
        backend_options: Options required by different backends.
        debug: Enable verbose logging. Disclaimer: this option must change in
            the near future.
        online_prepare: Compose QNN graph on device if set to True
        saver: Instead of compiling the model, run QNN Saver. Please check
            documents of Qualcomm AI Engine Direct SDK. This feature is usually
            for debugging purpose.
        dump_intermediate_outputs: If tensor dump is enabled, all intermediate tensors output will be dumped.
            This option exists for debugging accuracy issues
        profile: Enable profile the performance of per operator.
            Note that for now only support kProfileDetailed to
            profile the performance of each operator with cycle unit.
        shared_buffer: Enables usage of shared buffer between application
            and backend for graph I/O.
        is_from_context_binary: True if current graph comes from pre-built context binary.
        op_package_options: Optional structure to specify op packages
            loaded and used by the backend.
        use_mha2sha: This experimental parameter is used to decide whether to enable multi-head attention to single-head attention pass, aiming to reduce time consumption in AOT and improve performance on HTP.

    Returns:
        List[CompileSpec]: Compiler specs for Qualcomm AI Engine Direct.

    Raises:
        ValueError: The value QcomChipset is currently not supported.
        ValueError: Confliction between compiler specs.
    """
    _supported_soc_models = {soc_model.value for soc_model in QcomChipset}
    if soc_model not in _supported_soc_models:
        raise ValueError(f"unknown SoC model for QNN: {soc_model}")

    if profile and dump_intermediate_outputs:
        warnings.warn(
            "It is not recommended to turn on both profiling and dump_intermediate_outputs the same time"
            ", because dump_intermediate_outputs will cause performance drop.",
            stacklevel=1,
        )

    qnn_executorch_options = QnnExecuTorchOptions(
        _soc_info_table[soc_model], backend_options
    )
    qnn_executorch_options.log_level = (
        QnnExecuTorchLogLevel.kLogLevelDebug
        if debug
        else QnnExecuTorchLogLevel.kLogLevelError
    )

    qnn_executorch_options.dump_intermediate_outputs = dump_intermediate_outputs

    if saver:
        qnn_executorch_options.library_path = "libQnnSaver.so"
        qnn_executorch_options.saver = True
        qnn_executorch_options.saver_output_dir = "saver_output"

    if optrace:
        qnn_executorch_options.profile_level = QnnExecuTorchProfileLevel.kProfileOptrace
    elif profile:
        qnn_executorch_options.profile_level = (
            QnnExecuTorchProfileLevel.kProfileDetailed
        )
    else:
        qnn_executorch_options.profile_level = QnnExecuTorchProfileLevel.kProfileOff

    if (
        online_prepare
        and backend_options.backend_type == QnnExecuTorchBackendType.kHtpBackend
        and backend_options.htp_options.use_multi_contexts
    ):
        raise ValueError(
            "'use_multi_context' could not function in online prepare mode, "
            "please set 'online_prepare' to False"
        )

    qnn_executorch_options.shared_buffer = shared_buffer
    qnn_executorch_options.online_prepare = online_prepare
    qnn_executorch_options.is_from_context_binary = is_from_context_binary

    if op_package_options and len(op_package_options.op_package_infos) > 0:
        qnn_executorch_options.op_package_options = op_package_options

    qnn_executorch_options.use_mha2sha = use_mha2sha

    return [
        CompileSpec(QCOM_QNN_COMPILE_SPEC, option_to_flatbuffer(qnn_executorch_options))
    ]


def get_soc_to_arch_map():
    return {
        "SA8295": HtpArch.V68,
        "SM8350": HtpArch.V68,
        "SM8450": HtpArch.V69,
        "SM8475": HtpArch.V69,
        "SM8550": HtpArch.V73,
        "SA8255": HtpArch.V73,
        "SM8650": HtpArch.V75,
        "SM8750": HtpArch.V79,
        "SM8850": HtpArch.V81,
        "SSG2115P": HtpArch.V73,
        "SSG2125P": HtpArch.V73,
        "SXR1230P": HtpArch.V73,
        "SXR2230P": HtpArch.V69,
        "SXR2330P": HtpArch.V79,
        "QCS9100": HtpArch.V73,
        "SAR2230P": HtpArch.V81,
        "SW6100": HtpArch.V81,
        "QCM6490": HtpArch.V68,
        "SM8845": HtpArch.V81,
    }


def get_soc_to_chipset_map():
    return {
        "SA8295": QcomChipset.SA8295,
        "SM8350": QcomChipset.SM8350,
        "SM8450": QcomChipset.SM8450,
        "SM8475": QcomChipset.SM8475,
        "SM8550": QcomChipset.SM8550,
        "SA8255": QcomChipset.SA8255,
        "SM8650": QcomChipset.SM8650,
        "SM8750": QcomChipset.SM8750,
        "SM8850": QcomChipset.SM8850,
        "SSG2115P": QcomChipset.SSG2115P,
        "SSG2125P": QcomChipset.SSG2125P,
        "SXR1230P": QcomChipset.SXR1230P,
        "SXR2230P": QcomChipset.SXR2230P,
        "SXR2330P": QcomChipset.SXR2330P,
        "QCS9100": QcomChipset.QCS9100,
        "SAR2230P": QcomChipset.SAR2230P,
        "SW6100": QcomChipset.SW6100,
        "QCM6490": QcomChipset.QCM6490,
        "SM8845": QcomChipset.SM8845,
    }


def show_nn_module_stack_for_quant_recipe(gm: torch.fx.GraphModule, supported_ops):
    """
    Print a quick preview of op targets and module stack.

    Use this to inspect the FX graph and identify module stack, which helps you craft regex or op-target for quantization recipe.

    """

    module_metadata = {}
    for node in gm.graph.nodes:
        target = node.target
        deepest_module = None
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            deepest_module = list(node.meta["nn_module_stack"].values())[-1][0]
        if node.target in supported_ops:
            module_metadata.setdefault((target, deepest_module), []).append(node)

    table_rows = []
    for (target, module_stack), nodes in module_metadata.items():
        node_names = ", ".join([node.name for node in nodes])
        table_rows.append([str(target), module_stack, node_names])

    print(
        tabulate(
            table_rows, headers=["Op Target", "Module Stack", "Nodes"], tablefmt="grid"
        )
    )


def tag_quant_io(gm: torch.fx.GraphModule, get_quant_io_dtype_fn: Callable):
    """
    Tag io nodes which get/output quantized tensor. No need to insert q/dq in qnn_preprocess
    """
    for node in gm.graph.nodes:
        if dtype := get_quant_io_dtype_fn(node):
            node.meta[QCOM_QUANTIZED_IO] = dtype


def rewrite_prepared_observer(
    graph_module: torch.fx.GraphModule, name_obs_dict: Dict[str, torch.nn.Module]
):
    """
    Rewrite the observer of the specified observer module name in the graph_module.

    Example:
    Consider the following graph_module after prepare_pt2e:
    gm = prepare_pt2e(gm)
    print(gm)

    GraphModule(
      (activation_post_process_0): MinMaxObserver(min_val=inf, max_val=-inf)
      (activation_post_process_1): MinMaxObserver(min_val=inf, max_val=-inf)
      (activation_post_process_2): MinMaxObserver(min_val=inf, max_val=-inf)
      (activation_post_process_3): MinMaxObserver(min_val=inf, max_val=-inf)
    )

    new_observer = observer.FixedQParamsObserver(
        scale=0.125,
        zero_point=42,
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,
        qscheme=torch.per_tensor_affine,
    )

    Calling rewrite_prepared_observer(gm, {"activation_post_process_0": new_observer})
    is equivalent to:
    gm.activation_post_process_0 = new_observer

    Note:
    If the rewritten observer is a SharedQuantizationSpec, all other shared observers will also be rewritten.
    """
    module_name_list = defaultdict(list)
    for name, module in graph_module.named_modules(remove_duplicate=False):
        module_name_list[module].append(name)

    for name, new_observer in name_obs_dict.items():
        old_module = getattr(graph_module, name, None)

        if not old_module:
            print(
                f"[WARNING], No observer named as {name} found, please check the moudle name"
            )
            continue
        for target_name in module_name_list[old_module]:
            setattr(graph_module, target_name, new_observer)


def get_sdk_build_id():
    htp_library_path = (
        os.environ.get("QNN_SDK_ROOT", None) + "/lib/x86_64-linux-clang/libQnnHtp.so"
    )
    # The GetQnnSdkBuildId API can be used without needing to create a backend first, so it works regardless of which backend is used.
    sdk_build_id = PyQnnManagerAdaptor.GetQnnSdkBuildId(htp_library_path)
    return sdk_build_id


def is_qnn_sdk_version_less_than(target_version):
    current_version = get_sdk_build_id()

    match = re.search(r"v(\d+)\.(\d+)", current_version)
    if match:
        current_major, current_minor = map(int, match.groups()[:2])
    else:
        raise ValueError(
            f"Failed to get current major and minor version from QNN sdk Build id {current_version}"
        )

    target_major, target_minor = map(int, target_version.split(".")[:2])

    return current_major == target_major and current_minor < target_minor
