# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import collections
import copy
import os
import subprocess
import tempfile
import unittest
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from executorch import exir
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.quantizer.quantizer import (
    get_16a4w_qnn_ptq_config,
    get_default_16bit_qnn_ptq_config,
    QnnQuantizer,
    QuantDtype,
)
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import capture_program
from executorch.examples.qualcomm.scripts.utils import SimpleADB

from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.program._program import ExecutorchProgram
from executorch.sdk import generate_etrecord
from executorch.sdk.inspector import Inspector
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


def generate_context_binary(
    module: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    quantized: bool,
    artifact_dir: str,
):
    # we also expect clang showing in PATH or context may fail to generate
    qnn_sdk = os.environ.get("QNN_SDK_ROOT", None)
    ndk = os.environ.get("ANDROID_NDK_ROOT", None)
    assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
    assert ndk, "ANDROID_NDK_ROOT was not found in environment variable"

    inputs_tup = tuple(inputs.values())
    jit_module = torch.jit.trace(module, inputs_tup)
    torch.jit.save(jit_module, f"{artifact_dir}/jit_module.pt")

    # input data
    if quantized:
        input_list = []
        for name, data in inputs.items():
            file_name = f"{artifact_dir}/{name}.raw"
            data.detach().numpy().tofile(file_name)
            input_list.append(file_name)

        with open(f"{artifact_dir}/input_list.txt", "w") as f:
            f.write(" ".join(input_list))

    # flow of qnn tools
    target = "x86_64-linux-clang"
    inputs_str = [
        f"-d '{k}' {str(tuple(v.shape)).replace(' ', '')[1:-1]}"
        for k, v in inputs.items()
    ]
    cmds = [
        # setup qnn env
        f"source {qnn_sdk}/bin/envsetup.sh;"
        # qnn-pytorch-converter
        f"{qnn_sdk}/bin/{target}/qnn-pytorch-converter",
        f"-i {artifact_dir}/jit_module.pt",
        *inputs_str,
        f"--input_list {artifact_dir}/input_list.txt" if quantized else "",
        "--preserve_io",
        f"-o {artifact_dir}/model.cpp;",
        # qnn-model-lib-generator
        f"{qnn_sdk}/bin/{target}/qnn-model-lib-generator",
        f"-c {artifact_dir}/model.cpp",
        f"-t {target}",
        "-l model",
        f"-o {artifact_dir}/model_libs;",
        # qnn-context-binary-generator
        f"{qnn_sdk}/bin/{target}/qnn-context-binary-generator",
        f"--model {artifact_dir}/model_libs/{target}/libmodel.so",
        f"--backend {qnn_sdk}/lib/{target}/libQnnHtp.so",
        "--binary_file model_ctx",
        f"--output_dir {artifact_dir};",
    ]
    result = subprocess.run(
        " ".join(cmds),
        shell=True,
        executable="/bin/bash",
        capture_output=True,
    )
    assert os.path.isfile(f"{artifact_dir}/model_ctx.bin"), print(result.stderr)


class TestQNN(unittest.TestCase):
    rtol: float = 0
    atol: float = 0
    host: Literal = ""
    device: Literal = ""
    build_folder: Literal = ""
    model: QcomChipset = None
    compiler_specs: List[CompileSpec] = None
    arch_table = {
        "SM8650": QcomChipset.SM8650,
        "SM8550": QcomChipset.SM8550,
        "SM8475": QcomChipset.SM8475,
        "SM8450": QcomChipset.SM8450,
    }
    error_only = False
    ip = "localhost"
    port = 8080
    executorch_root: Literal = ""
    artifact_dir: Literal = ""
    image_dataset: Literal = ""
    pretrained_weight: Literal = ""
    enable_profile: bool = False
    online_prepare: bool = False
    use_8a8w: str = "8a8w"
    use_16a16w: str = "16a16w"
    use_16a4w: str = "16a4w"
    shared_buffer: bool = False

    def _assert_outputs_equal(self, model_output, ref_output):
        self.assertTrue(len(ref_output) == len(model_output))
        for i in range(len(ref_output)):
            self.assertTrue(
                torch.allclose(
                    model_output[i], ref_output[i], atol=self.atol, rtol=self.rtol
                ),
                msg=f"ref_output:\n{ref_output[i]}\n\nmodel_output:\n{model_output[i]}",
            )

    def _save_model_and_expected_output(
        self,
        module: torch.nn.Module,
        buffer: exir.ExirExportedProgram,
        inputs: Tuple[torch.Tensor],
        dir_name: Literal,
    ) -> None:
        # Save the input data list to be executed
        input_list = ""
        for idx, _ in enumerate(inputs):
            input_name = f"input_0_{idx}.raw"
            input_list += input_name + " "
        input_list = input_list.strip() + "\n"

        ref_output = module(*inputs)

        # Save the expected output data to be verified
        ref_outputs = []
        if isinstance(ref_output, collections.OrderedDict):
            ref_outputs.append(ref_output["out"].detach())
        elif isinstance(ref_output, tuple):
            for output in ref_output:
                ref_outputs.append(output.detach())
        else:
            ref_outputs.append(ref_output.detach())

        pte_fname = f"{dir_name}/qnn_executorch_test.pte"
        with open(pte_fname, "wb") as file:
            file.write(buffer)

        return input_list, ref_outputs, pte_fname

    def verify_output(
        self,
        module: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        executorch_prog: ExecutorchProgram | LoweredBackendModule,
        etrecord_path: str = "etrecord.bin",
        expected_profile_events: int = -1,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            buffer = (
                executorch_prog.buffer
                if isinstance(executorch_prog, ExecutorchProgram)
                else executorch_prog.buffer()
            )
            (
                input_list,
                ref_outputs,
                pte_fname,
            ) = self._save_model_and_expected_output(
                module,
                buffer,
                sample_inputs,
                tmp_dir,
            )

            device_output_dir = f"{tmp_dir}/outputs"
            device_outputs = []
            etdump_path = f"{tmp_dir}/etdump.etdp"

            def post_process():
                for i, f in enumerate(sorted(os.listdir(device_output_dir))):
                    filename = os.path.join(device_output_dir, f)
                    output = np.fromfile(filename, dtype=ref_outputs[i].numpy().dtype)
                    output = torch.from_numpy(output).reshape(ref_outputs[i].shape)
                    device_outputs.append(output)

            def validate_profile():
                inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)
                self.assertTrue(
                    len(inspector.to_dataframe().index) == expected_profile_events
                )

            adb = SimpleADB(
                qnn_sdk=os.getenv("QNN_SDK_ROOT"),
                build_path=self.build_folder,
                pte_path=pte_fname,
                workspace="/data/local/tmp/qnn_executorch_test",
                device_id=self.device,
                host_id=self.host,
                soc_model=self.model,
                error_only=self.error_only,
            )
            adb.push(inputs=[sample_inputs], input_list=input_list)
            adb.execute()
            adb.pull(output_path=tmp_dir, callback=post_process)
            self._assert_outputs_equal(device_outputs, ref_outputs)

            if expected_profile_events != -1:
                adb.pull_etdump(etdump_path, callback=validate_profile)

    def lower_module_and_test_output(
        self,
        module: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        expected_partitions: int = 1,
        expected_profile_events: int = -1,
        assert_output_equal: bool = True,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        qnn_partitioner = QnnPartitioner(
            self.compiler_specs, skip_node_id_set, skip_node_op_set
        )
        delegated_program = capture_program(module, sample_inputs)

        # this is needed for the ETRecord as lowering modifies the graph in-place
        edge_copy = copy.deepcopy(delegated_program)

        delegated_program.exported_program = to_backend(
            delegated_program.exported_program, qnn_partitioner
        )
        exec_prog = delegated_program.to_executorch(
            exir.ExecutorchBackendConfig(
                extract_delegate_segments=False,
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner.
                # Therefore, won't want to pre-allocate
                # by memory manager in runtime.
                memory_planning_pass=MemoryPlanningPass(
                    memory_planning_algo="greedy",
                    alloc_graph_input=not self.shared_buffer,
                    alloc_graph_output=not self.shared_buffer,
                ),
            )
        )

        # Assert the backend name is qnn
        self.assertEqual(
            len(exec_prog.program.execution_plan[0].delegates),
            expected_partitions,
        )
        for i in range(expected_partitions):
            self.assertEqual(
                exec_prog.program.execution_plan[0].delegates[i].id,
                QnnBackend.__name__,
            )

        etrecord_path = "etrecord.bin"
        if self.enable_profile:
            generate_etrecord(etrecord_path, edge_copy, exec_prog)

        # Check numerics
        if assert_output_equal or expected_profile_events != -1:
            self.verify_output(
                module, sample_inputs, exec_prog, etrecord_path, expected_profile_events
            )

    def get_qdq_module(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        is_conv_per_channel: Optional[bool] = True,
        is_linear_per_channel: Optional[bool] = False,
        custom_quant_annotations: Tuple[Callable] = (),
        quant_dtype: QuantDtype = QuantDtype.use_8a8w,
    ) -> torch.fx.GraphModule:
        m = torch._export.capture_pre_autograd_graph(module, inputs)

        quantizer = QnnQuantizer()
        quantizer.add_custom_quant_annotations(custom_quant_annotations)
        quantizer.set_per_channel_conv_quant(is_conv_per_channel)
        quantizer.set_per_channel_linear_quant(is_linear_per_channel)

        if quant_dtype == QuantDtype.use_8a8w:
            pass  # default setting
        elif quant_dtype == QuantDtype.use_16a16w:
            quantizer.add_16bit_quant_ops(quantizer.SUPPORTED_OPS)
            quantizer.set_bit16_op_quant_config(get_default_16bit_qnn_ptq_config())
        elif quant_dtype == QuantDtype.use_16a4w:
            quantizer.add_16bit_quant_ops(quantizer.SUPPORTED_OPS)
            quantizer.set_bit16_op_quant_config(get_16a4w_qnn_ptq_config())
            quantizer.set_per_channel_weight_dtype(weight_dtype_for_16bit_act="int4")
        else:
            raise AssertionError(f"No support for QuantDtype {quant_dtype}.")

        prepared = prepare_pt2e(m, quantizer)
        prepared(*inputs)
        quantized_module = convert_pt2e(prepared)
        nodes = {node.target for node in quantized_module.graph.nodes}
        q_and_dq = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        }
        self.assertTrue(nodes.intersection(q_and_dq))
        return quantized_module

    def split_graph(self, graph_module: torch.fx.GraphModule, division: int):
        class SplitGraph(ExportPass):
            """
            Split graph based on number of nodes.
            """

            def __init__(self, shares):
                super().__init__()
                self.shares = shares

            def _insert_clone(
                self, graph_module: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                num_graph_nodes = 0
                for node in graph_module.graph.nodes:
                    num_graph_nodes += 1 if node.op == "call_function" else 0

                    if num_graph_nodes % self.shares != 0 or node.op != "call_function":
                        continue

                    with graph_module.graph.inserting_after(node):
                        users = list(node.users.keys())
                        inserted_node = graph_module.graph.create_node(
                            "call_function",
                            exir_ops.edge.aten.clone.default,
                            (node,),
                        )
                        inserted_node.meta["val"] = node.meta["val"]
                        for user in users:
                            user.replace_input_with(node, inserted_node)

            def call(self, graph_module: torch.fx.GraphModule):
                self._insert_clone(graph_module)
                graph_module.recompile()

        num_graph_nodes = 0
        for node in graph_module.graph.nodes:
            num_graph_nodes += 1 if node.op == "call_function" else 0

        SplitGraph(-(num_graph_nodes // -division))(graph_module)
