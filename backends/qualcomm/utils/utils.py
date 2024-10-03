# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor

import executorch.exir as exir

import torch

from executorch.backends.qualcomm.builders.node_visitor import (
    QNN_QUANT_TYPE_MAP,
    QNN_TENSOR_TYPE_MAP,
)
from executorch.backends.qualcomm.builders.qnn_constants import OpContextLoader
from executorch.backends.qualcomm.passes.annotate_and_quant_scalar import (
    AnnotateAndQuantScalar,
)
from executorch.backends.qualcomm.passes.annotate_decomposed import AnnotateDecomposed
from executorch.backends.qualcomm.passes.annotate_quant_attrs import AnnotateQuantAttrs
from executorch.backends.qualcomm.passes.convert_binary_op_with_scalar import (
    ConvertBinaryOpsWithScalar,
)
from executorch.backends.qualcomm.passes.convert_bmm_to_matmul import ConvertBmmToMatmul
from executorch.backends.qualcomm.passes.convert_interpolate_with_upsample2d import (
    ConvertInterpolateWithUpsample2D,
)
from executorch.backends.qualcomm.passes.convert_prelu import ConvertPReLU
from executorch.backends.qualcomm.passes.convert_to_linear import ConvertToLinear
from executorch.backends.qualcomm.passes.fold_qdq import FoldQDQ
from executorch.backends.qualcomm.passes.i64_to_i32 import I64toI32
from executorch.backends.qualcomm.passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.passes.recompose_pixel_unshuffle import (
    RecomposePixelUnshuffle,
)
from executorch.backends.qualcomm.passes.recompose_rms_norm import RecomposeRmsNorm
from executorch.backends.qualcomm.passes.remove_redundancy import RemoveRedundancy
from executorch.backends.qualcomm.passes.replace_index_put_input import (
    ReplaceIndexPutInput,
)
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    _soc_info_table,
    QcomChipset,
    QnnExecuTorchBackendOptions,
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpBackendOptions,
    QnnExecuTorchHtpPerformanceMode,
    QnnExecuTorchHtpPrecision,
    QnnExecuTorchLogLevel,
    QnnExecuTorchOptions,
    QnnExecuTorchProfileLevel,
)
from executorch.backends.qualcomm.serialization.qnn_compile_spec_serialize import (
    convert_to_flatbuffer,
    convert_to_option,
)
from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

from executorch.exir import ExirExportedProgram
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.program._program import _get_updated_graph_signature
from torch._decomp import core_aten_decompositions as torch_core_aten_decompositions
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
        _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
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
            x = x.unsqueeze(-1) if rank == 3 else x.reshape(1, *x.shape, 1)
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


def canonicalize_program(
    exported_program: ExportedProgram | List[LoweredBackendModule],
    custom_buffer_size=None,
):
    # check if user specifies to use multi_contexts
    # this is a generic approach in case there exists multiple backends
    def get_program_info(program):
        def process_exported_program(prog):
            max_sf_buf_size, module_map = 0, {}
            for _, m in prog.graph_module._modules.items():
                # currently only 1 compile spec is expected in each partition
                options = convert_to_option(m.compile_specs[0].value)
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
            spill_fill_size = (
                qnn_mgr.GetSpillFillBufferSize()
                if custom_buffer_size is None
                else custom_buffer_size
            )
            qnn_mgr.Destroy()
            return spill_fill_size, {
                module: convert_to_option(module.compile_specs[0].value)
            }

        dispatch = {
            ExportedProgram: process_exported_program,
            LoweredBackendModule: process_lowered_module,
        }
        return dispatch[type(program)](program)

    def update_program(max_sf_buf_size, module_map):
        def set_spec(module, options):
            spec = CompileSpec(QCOM_QNN_COMPILE_SPEC, convert_to_flatbuffer(options))
            if isinstance(module, ExportedProgram):
                module.compile_specs[0] = spec
            else:
                module._compile_specs[0] = spec

        for module, options in module_map.items():
            options.backend_options.htp_options.max_sf_buf_size = max_sf_buf_size
            set_spec(module, options)

    if isinstance(exported_program, list):
        max_sf_size, modules_map = 0, {}
        for prog in exported_program:
            max_sf_buf_size, module_map = get_program_info(prog)
            max_sf_size = max(max_sf_size, max_sf_buf_size)
            modules_map.update(module_map)
        update_program(max_sf_size, modules_map)
    else:
        update_program(*get_program_info(exported_program))


def get_decomp_table() -> Dict[torch._ops.OperatorBase, Callable]:
    source_decompositions = torch_core_aten_decompositions()
    # The below super ops are supported by QNN
    remove_decompositions = [
        torch.ops.aten.pixel_shuffle.default,
        torch.ops.aten.pixel_unshuffle.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten._safe_softmax.default,
    ]

    for key in remove_decompositions:
        source_decompositions.pop(key)

    return source_decompositions


def _transform(edge_program: ExportedProgram) -> None:
    # currently ExirExportedProgram.transform does not accept
    # changes of input number which was caused by FoldQDQ
    # apply passes one by one here to avoid IR capture failure
    graph_module = edge_program.graph_module
    RemoveRedundancy()(graph_module)
    RecomposePixelUnshuffle()(graph_module)
    RecomposeRmsNorm()(graph_module)
    ConvertToLinear()(graph_module)
    ConvertPReLU(edge_program)(graph_module)
    ConvertBmmToMatmul()(graph_module)
    ConvertInterpolateWithUpsample2D()(graph_module)
    I64toI32(edge_program)(graph_module)
    AnnotateQuantAttrs(edge_program)(graph_module)
    AnnotateAndQuantScalar(edge_program)(graph_module)
    AnnotateDecomposed(edge_program)(graph_module)
    FoldQDQ()(graph_module)
    LayoutTransform(edge_program)(graph_module)
    ReplaceIndexPutInput(edge_program)(graph_module)

    # Since QDQ nodes are stripped, update graph signature again to validate program
    edge_program._graph_signature = _get_updated_graph_signature(
        edge_program.graph_signature,
        edge_program.graph_module,
    )
    edge_program._validate()


def capture_program(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
) -> exir.ExirExportedProgram:
    ep = torch.export.export(module, inputs)
    decomposed_ep = ep.run_decompositions(get_decomp_table())

    # We choose call_operator by target in ConvertBinaryOpsWithScalar
    # because it is the same source_fn_stack for MultiheadAttention
    # TODO: Should modify the scalar op in the op builder instead of
    #       using transformation
    core_ep = ExirExportedProgram(decomposed_ep, False)
    core_ep.transform(ConvertBinaryOpsWithScalar())
    edge_ep = core_ep.to_edge(qnn_edge_config())
    _transform(edge_ep.exported_program)
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


def _canonicalize_graph_with_lowered_module(gm, subgm_tag, ptn):
    from executorch.exir.backend.backend_api import to_backend

    # return lowered program for user to debug
    exported_progs = []
    # partition each submodule which went through convert_pt2e
    for node in gm.graph.nodes:
        if node.op == "call_module" and subgm_tag in node.name:
            # obtain sample inputs through meta
            subgm_input = [
                torch.ones(arg.meta["val"].shape, dtype=arg.meta["val"].dtype)
                for arg in node.args
            ]
            # program meets QNN backend requirement
            sub_prog = capture_program(gm.get_submodule(node.name), tuple(subgm_input))
            # start lowering with given partitioner
            exported_progs.append(to_backend(sub_prog.exported_program, ptn))
            # replace submodule with lowered module
            gm.set_submodule(
                node.name,
                exported_progs[-1].graph_module,
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
    return gm, exported_progs


def skip_annotation(
    nn_module: torch.nn.Module,
    quantizer,
    partitioner,
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
        partitioner (QnnPartitioner): Instance of QnnPartitioner.
        sample_input ((torch.Tensor, ...)): Sample input tensors for graph exporting.
        calibration_cb (callable): Callback function for user-defined calibration.
        fp_node_id_set ({str, ...}): Set of operator names to be left in fp precision.
        fp_node_op_set ({torch.ops.aten.xxx, ...}): Set of operator targets to be left in fp precision.
        fallback_to_cpu (bool): Whether to lower skipped nodes to fp16 or not.

    Returns:
        exported_programs: List of programs lowered to QnnBackend (quantized graphs only).
    """
    from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
        QnnExecuTorchHtpPrecision,
    )
    from executorch.backends.qualcomm.serialization.qnn_compile_spec_serialize import (
        convert_to_option,
    )
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

    def prepare_subgm(subgm, subgm_name):
        # prepare current submodule for quantization annotation
        subgm_prepared = prepare_pt2e(subgm, quantizer)
        # overwrite this attribute or name will be set to "GraphModule"
        # we could not identify each submodule if action is not performed
        subgm_prepared.__class__.__name__ = subgm_name
        return subgm_prepared

    fp_node_id_set = fp_node_id_set if fp_node_id_set is not None else set()
    fp_node_op_set = fp_node_op_set if fp_node_op_set is not None else set()
    graph_module = torch.export.export(nn_module, sample_input).module()
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
                node.name, convert_pt2e(graph_module.get_submodule(node.name))
            )
    # canonicalize graph for lowering again
    graph_module, exported_progs = _canonicalize_graph_with_lowered_module(
        gm=graph_module,
        subgm_tag=subgm_tag,
        ptn=partitioner,
    )

    if not fallback_to_cpu:
        try:
            from executorch.exir.backend.partitioner import DelegationSpec

            # change HTP compiler spec for hardware to enable fp16
            qnn_option = generate_qnn_executorch_option(
                partitioner.compiler_specs_snapshot
            )
            compile_option = convert_to_option(qnn_option)
            htp_options = compile_option.backend_options.htp_options
            htp_options.precision = QnnExecuTorchHtpPrecision.kHtpFp16
            partitioner.delegation_spec = DelegationSpec(
                "QnnBackend",
                [
                    CompileSpec(
                        QCOM_QNN_COMPILE_SPEC, convert_to_flatbuffer(compile_option)
                    )
                ],
            )
        except:
            print(
                "Failed to change HTP compiler spec with 'use_fp16' as True,"
                " skipped operators will fallback to cpu,"
            )
            return graph_module, exported_progs

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
        graph_module, exported_progs_fp = _canonicalize_graph_with_lowered_module(
            gm=graph_module,
            subgm_tag=subgm_tag,
            ptn=partitioner,
        )
        exported_progs.extend(exported_progs_fp)

    return graph_module, exported_progs


def from_context_binary(
    ctx_path: str, op_name: str, soc_model: QcomChipset = QcomChipset.SM8650
):
    def implement_op(custom_op, op_name, outputs):
        @torch.library.impl(
            custom_op, str(op_name), dispatch_key="CompositeExplicitAutograd"
        )
        def op_impl(inputs: List[torch.Tensor]):
            return tuple(
                torch.zeros(tuple(v.shape), device="meta", dtype=v.dtype)
                for v in outputs.values()
            )

    def build_graph(inputs, outputs):
        # custom op declaration
        inputs_str = "Tensor[] inputs"
        func_proto = f"{op_name}({inputs_str}) -> Any"
        custom_op = Library(OpContextLoader.namespace, "FRAGMENT")
        custom_op.define(func_proto)
        # custom op implementation
        implement_op(custom_op, op_name, outputs)

        # model architecture mimicking context binary
        class Model(torch.nn.Module):
            def forward(self, *inputs):
                return getattr(
                    getattr(torch.ops, OpContextLoader.namespace), op_name
                ).default(inputs)

        model = Model()
        prog = torch.export.export(model, tuple(inputs.values()))
        # bookkeeping for variables' life cycle
        return {
            "custom_op": custom_op,
            "custom_module": model,
            "edge_program": prog,
        }

    def build_tensor(tensors, dtype_map):
        ret = OrderedDict()
        for t in tensors:
            dtype = t.GetDataType()
            dtype_torch = dtype_map.get(dtype, None)
            assert dtype_torch is not None, f"unknown qnn data type {dtype}"
            ret[t.GetName()] = torch.zeros(tuple(t.GetDims()), dtype=dtype_torch)

        return ret

    with open(ctx_path, "rb") as f:
        ctx_bin = f.read()
    # dummy compiler spec would be fine, since we're not compiling
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_model,
        backend_options=backend_options,
        is_from_context_binary=True,
    )
    # get context-binary io tensor info through qnn manager
    qnn_mgr = PyQnnManagerAdaptor.QnnManager(
        generate_qnn_executorch_option(compiler_specs), ctx_bin
    )
    assert qnn_mgr.Init().value == 0, "failed to load context binary"
    qnn_mgr.AllocateTensor()
    dtype_map = {}
    for type_map in (QNN_QUANT_TYPE_MAP, QNN_TENSOR_TYPE_MAP):
        for k, v in type_map.items():
            dtype_map.setdefault(v, k)
    inputs = build_tensor(qnn_mgr.GetGraphInputs(), dtype_map)
    outputs = build_tensor(qnn_mgr.GetGraphOutputs(), dtype_map)
    qnn_mgr.Destroy()
    # generate graph specific for loading context
    bundle_prog = build_graph(inputs, outputs)
    bundle_prog.update({"inputs": inputs, "outputs": outputs})
    for n in bundle_prog["edge_program"].graph_module.graph.nodes:
        if op_name in n.name:
            n.meta[OpContextLoader.meta_ctx_bin] = ctx_bin
            break
    return bundle_prog


def draw_graph(title, path, graph_module: torch.fx.GraphModule):
    graph = passes.graph_drawer.FxGraphDrawer(graph_module, title)
    with open(f"{path}/{title}.svg", "wb") as f:
        f.write(graph.get_dot_graph().create_svg())


def generate_qnn_executorch_option(
    compiler_specs: List[CompileSpec],
) -> bytes:
    for compiler_spec in compiler_specs:
        if compiler_spec.key == QCOM_QNN_COMPILE_SPEC:
            qnn_compile_spec_buffer = compiler_spec.value
        else:
            raise ValueError(f"unknown compiler spec key value: {compiler_spec.key}")
    return qnn_compile_spec_buffer


def generate_htp_compiler_spec(
    use_fp16: bool,
    use_dlbc: bool = False,
    use_multi_contexts: bool = False,
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
    shared_buffer: bool = False,
    is_from_context_binary: bool = False,
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
    qnn_executorch_options.graph_name = "executorch"
    qnn_executorch_options.log_level = (
        QnnExecuTorchLogLevel.kLogLevelDebug
        if debug
        else QnnExecuTorchLogLevel.kLogLevelWarn
    )

    qnn_executorch_options.dump_intermediate_outputs = dump_intermediate_outputs

    if saver:
        qnn_executorch_options.library_path = "libQnnSaver.so"

    if profile:
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

    return [
        CompileSpec(
            QCOM_QNN_COMPILE_SPEC, convert_to_flatbuffer(qnn_executorch_options)
        )
    ]
