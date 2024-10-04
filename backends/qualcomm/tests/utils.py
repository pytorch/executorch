# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import collections
import os
import subprocess
import tempfile
import unittest
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
import torch
import torchao
from executorch import exir
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
    QNNIntermediateDebugger,
)
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.quantizer.quantizer import ModuleQConfig, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DTYPE,
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)
from executorch.backends.qualcomm.utils.utils import (
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools import Inspector
from executorch.devtools.inspector._inspector_utils import TimeScale
from executorch.examples.qualcomm.utils import (
    generate_inputs,
    make_output_dir,
    make_quantizer,
    SimpleADB,
)

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.utils import get_delegates
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.program import ExecutorchProgram, ExecutorchProgramManager
from torch.fx.passes.infra.pass_base import PassResult
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)


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


def validate_context_binary(ctx_bin: bytes):
    qnn_sdk = os.environ.get("QNN_SDK_ROOT", None)
    assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"

    # flow of qnn tools
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/ctx.bin", "wb") as binary_file:
            binary_file.write(ctx_bin)

        target = "x86_64-linux-clang"
        cmds = [
            # qnn-context-binary-utility
            f"{qnn_sdk}/bin/{target}/qnn-context-binary-utility",
            "--context_binary",
            f"{tmp_dir}/ctx.bin",
            "--json_file",
            f"{tmp_dir}/ctx.json",
        ]
        result = subprocess.run(
            " ".join(cmds),
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )
        assert os.path.isfile(f"{tmp_dir}/ctx.json"), print(result.stderr)


class TestQNN(unittest.TestCase):
    rtol: float = 0
    atol: float = 0
    host: str = ""
    device: str = ""
    build_folder: str = ""
    model: QcomChipset = None
    compiler_specs: List[CompileSpec] = None
    chipset_table = get_soc_to_chipset_map()
    error_only = False
    ip = "localhost"
    port = 8080
    executorch_root: str = ""
    artifact_dir: str = ""
    image_dataset: str = ""
    qa_dataset: str = ""
    sentence_dataset: str = ""
    pretrained_weight: str = ""
    enable_profile: bool = False
    op_package_dir: str = ""
    target: str = ""
    model_name: str = ""
    backend: str = ""
    online_prepare: bool = False
    use_8a8w: str = "8a8w"
    use_16a16w: str = "16a16w"
    use_16a4w: str = "16a4w"
    oss_repo: str = ""
    shared_buffer: bool = False
    enable_x86_64: bool = False
    compile_only: bool = False
    pre_gen_pte: str = ""
    llama_artifacts: str = ""
    dump_intermediate_outputs: bool = False
    inference_speed: float = 0.0
    inference_speed_output_path = "outputs/inference_speed.txt"

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
        dir_name: str,
    ) -> None:
        ref_output = module(*inputs)

        # Save the expected output data to be verified
        ref_outputs = []
        if isinstance(ref_output, collections.OrderedDict):
            ref_outputs.append(ref_output["out"].detach())
        elif isinstance(ref_output, (list, tuple)):
            for output in ref_output:
                ref_outputs.append(output.detach())
        else:
            ref_outputs.append(ref_output.detach())

        pte_fname = f"{dir_name}/qnn_executorch_test.pte"
        with open(pte_fname, "wb") as file:
            file.write(buffer)

        return ref_outputs, pte_fname

    def get_backend_type(self):
        return getattr(QnnExecuTorchBackendType, f"k{self.backend.title()}Backend")

    def required_envs(self, conditions=None) -> bool:
        conditions = [] if conditions is None else conditions
        return all(
            [
                self.executorch_root,
                self.artifact_dir,
                *conditions,
            ]
        )

    def verify_output(  # noqa: C901
        self,
        module: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        executorch_prog: ExecutorchProgram | ExecutorchProgramManager,
        etrecord_path: str = "etrecord.bin",
        expected_profile_events: int = -1,
        expected_intermediate_events: int = -1,
        method_index: int = 0,
        input_encodings: Tuple = (),
        output_encodings: Tuple = (),
        check_io_shape: bool = False,
        op_package_paths: List[str] = None,
        extra_cmds: str = "",
        output_callback: Optional[Callable[[str], None]] = None,
        save_inference_speed: bool = False,
        expected_compared_events: int = -1,
        qnn_intermediate_debugger: QNNIntermediateDebugger = None,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (
                ref_outputs,
                pte_fname,
            ) = self._save_model_and_expected_output(
                module,
                executorch_prog.buffer,
                sample_inputs,
                tmp_dir,
            )

            output_dir = f"{tmp_dir}/outputs"
            outputs = []
            etdump_path = f"{tmp_dir}/etdump.etdp"
            debug_output_path = f"{tmp_dir}/debug_output.bin"

            def post_process():
                from torch.testing._internal.common_utils import (
                    torch_to_numpy_dtype_dict,
                )

                for i, f in enumerate(
                    sorted(f for f in os.listdir(output_dir) if f.endswith(".raw"))
                ):
                    enc = output_encodings[i] if len(output_encodings) != 0 else None
                    dtype = (
                        ref_outputs[i].numpy().dtype
                        if enc is None
                        else torch_to_numpy_dtype_dict[enc[QCOM_DTYPE]]
                    )
                    filename = os.path.join(output_dir, f)
                    output = np.fromfile(filename, dtype=dtype)
                    output = torch.from_numpy(output).reshape(ref_outputs[i].shape)
                    if enc is not None:
                        output = (
                            output.to(torch.float)
                            .sub(enc[QCOM_ZERO_POINT])
                            .mul(enc[QCOM_SCALE])
                        )
                    outputs.append(output)

            def validate_profile():
                inspector = Inspector(
                    etdump_path=etdump_path,
                    etrecord=etrecord_path,
                    source_time_scale=TimeScale.CYCLES,
                    target_time_scale=TimeScale.CYCLES,
                )
                self.assertTrue(
                    len(inspector.to_dataframe().index) >= expected_profile_events
                )

            def validate_intermediate_tensor():
                inspector = Inspector(
                    etdump_path=etdump_path, debug_buffer_path=debug_output_path
                )
                node_tensor_map = qnn_intermediate_debugger._match_tensors(
                    inspector=inspector, keep_qnn_layout=False
                )
                self.assertTrue(
                    len(node_tensor_map) == expected_compared_events,
                    msg=f"Unexpected number of compared events, expecting {expected_compared_events}, but has {len(node_tensor_map)} events.",
                )
                # Compare accuracy for each layer
                for _, value in node_tensor_map.items():
                    self._assert_outputs_equal(
                        value[0].to(torch.float32), value[1].to(torch.float32)
                    )
                for event_block in inspector.event_blocks:
                    if event_block.name == "Execute":
                        self.assertTrue(
                            len(event_block.events) == expected_intermediate_events,
                            msg=f"Unexpected number of intermediate events, expecting {expected_intermediate_events}, but has {len(event_block.events)} events.",
                        )

            processed_inputs = list(sample_inputs)
            for i, enc in enumerate(input_encodings):
                processed_inputs[i] = (
                    processed_inputs[i]
                    .div(enc[QCOM_SCALE])
                    .add(enc[QCOM_ZERO_POINT])
                    .round()
                    .to(enc[QCOM_DTYPE])
                )

            if self.enable_x86_64:
                generate_inputs(tmp_dir, "input_list.txt", [processed_inputs])
                make_output_dir(output_dir)

                target = "x86_64-linux-clang"
                qnn_sdk = os.environ.get("QNN_SDK_ROOT", None)
                assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"

                build_folder = self.build_folder
                if os.path.isabs(self.build_folder):
                    # obey user's opinion
                    pass
                else:
                    # ok, assuming the user give a relative path to cwd
                    build_folder = os.path.join(os.getcwd(), self.build_folder)

                cmd = [
                    # qnn_executor_runner
                    f"{build_folder}/examples/qualcomm/executor_runner/qnn_executor_runner",
                    "--model_path",
                    pte_fname,
                    "--input_list_path",
                    f"{tmp_dir}/input_list.txt",
                    "--output_folder_path",
                    output_dir,
                    "--method_index",
                    str(method_index),
                ]
                if expected_intermediate_events != -1:
                    cmd.append("--dump_intermediate_outputs")
                cmd += extra_cmds.split()

                if save_inference_speed:
                    cmd += [
                        "--performance_output_path",
                        self.inference_speed_output_path,
                    ]

                if check_io_shape:
                    shape_info = {
                        "input_shape": processed_inputs,
                        "output_shape": ref_outputs,
                    }
                    for name, tensors in shape_info.items():
                        with open(f"{tmp_dir}/{name}.txt", "w") as f:
                            for t in tensors:
                                f.write(str(tuple(t.shape)).strip("()") + "\n")
                        cmd.append(f"--{name}_path")
                        cmd.append(f"{tmp_dir}/{name}.txt")

                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = f"{qnn_sdk}/lib/{target}/:{build_folder}/lib"
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=tmp_dir,
                )

                if output_callback:
                    output_callback(proc)
                self.assertEqual(
                    proc.returncode,
                    0,
                    f"The process running qnn_executorch_runner return {proc.returncode}, "
                    "STDOUT=\n"
                    f"{proc.stdout}",
                )

                # Verify the outputs
                post_process()
                self._assert_outputs_equal(outputs, ref_outputs)

                # Verify the etdump
                if expected_profile_events != -1:
                    validate_profile()

                if expected_intermediate_events != -1:
                    validate_intermediate_tensor()

                if save_inference_speed:
                    with open(
                        f"{tmp_dir}/{self.inference_speed_output_path}", "r"
                    ) as f:
                        self.inference_speed = float(f.read())

            else:
                adb = SimpleADB(
                    qnn_sdk=os.getenv("QNN_SDK_ROOT"),
                    build_path=self.build_folder,
                    pte_path=pte_fname,
                    workspace="/data/local/tmp/qnn_executorch_test",
                    device_id=self.device,
                    host_id=self.host,
                    soc_model=self.model,
                    error_only=self.error_only,
                    dump_intermediate_outputs=(
                        True if expected_intermediate_events != -1 else False
                    ),
                    backend=self.get_backend_type(),
                    expected_input_shape=(
                        (tensor.shape for tensor in processed_inputs)
                        if check_io_shape
                        else None
                    ),
                    expected_output_shape=(
                        (tensor.shape for tensor in ref_outputs)
                        if check_io_shape
                        else None
                    ),
                    target=self.target,
                )
                adb.push(
                    inputs=[processed_inputs],
                    files=op_package_paths,
                )
                adb.extra_cmds += extra_cmds
                if save_inference_speed:
                    adb.extra_cmds += (
                        f" --performance_output_path {self.inference_speed_output_path}"
                    )
                adb.execute(method_index=method_index, output_callback=output_callback)
                adb.pull(output_path=tmp_dir, callback=post_process)
                self._assert_outputs_equal(outputs, ref_outputs)

                if expected_profile_events != -1:
                    adb.pull_etdump(etdump_path, callback=validate_profile)

                if expected_intermediate_events != -1:
                    adb.pull_debug_output(
                        etdump_path,
                        debug_output_path,
                        callback=validate_intermediate_tensor,
                    )
                if save_inference_speed:
                    with open(
                        f"{tmp_dir}/{self.inference_speed_output_path}", "r"
                    ) as f:
                        self.inference_speed = float(f.read())

    def lower_module_and_test_output(
        self,
        module: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        expected_partitions: int = 1,
        expected_profile_events: int = -1,
        expected_intermediate_events: int = -1,
        expected_compared_events: int = -1,
        assert_output_equal: bool = True,
        passes_job: Optional[OrderedDict] = None,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
        skip_mutable_buffer: bool = False,
        dynamic_shapes: Dict = None,
        extra_cmds: str = "",
        output_callback: Optional[Callable[[str], None]] = None,
        save_inference_speed: bool = False,
    ):
        delegated_program = to_edge_transform_and_lower_to_qnn(
            module,
            sample_inputs,
            self.compiler_specs,
            dynamic_shapes=dynamic_shapes,
            passes_job=passes_job,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            skip_mutable_buffer=skip_mutable_buffer,
            generate_etrecord=self.enable_profile,
        )

        qnn_intermediate_debugger = None
        if expected_intermediate_events != -1:
            lowered_module_nodes = get_delegates(
                delegated_program.exported_program().graph
            )
            assert len(lowered_module_nodes) == 1, "Length not correct"

            lowered_module_node = lowered_module_nodes[0]
            lower_module = getattr(
                delegated_program.exported_program().graph_module,
                lowered_module_node.name,
            )
            edge_module = lower_module.original_module.module()

            qnn_intermediate_debugger = QNNIntermediateDebugger()
            qnn_intermediate_debugger.set_edge_module(edge_module=edge_module)
            qnn_intermediate_debugger.intermediate_output_module(*sample_inputs)

        exec_prog = delegated_program.to_executorch(
            exir.ExecutorchBackendConfig(
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner.
                # Therefore, won't want to pre-allocate
                # by memory manager in runtime.
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=not self.shared_buffer,
                    alloc_graph_output=not self.shared_buffer,
                ),
            )
        )

        # Assert the backend name is qnn
        self.assertEqual(
            len(exec_prog.executorch_program.execution_plan[0].delegates),
            expected_partitions,
        )
        for i in range(expected_partitions):
            self.assertEqual(
                exec_prog.executorch_program.execution_plan[0].delegates[i].id,
                QnnBackend.__name__,
            )

        etrecord_path = "etrecord.bin"
        if self.enable_profile:
            exec_prog.get_etrecord().save(etrecord_path)
        # Check numerics
        if (
            assert_output_equal
            or expected_profile_events != -1
            or expected_intermediate_events != -1
        ):
            self.verify_output(
                module=module,
                sample_inputs=sample_inputs,
                executorch_prog=exec_prog,
                etrecord_path=etrecord_path,
                expected_profile_events=expected_profile_events,
                expected_intermediate_events=expected_intermediate_events,
                extra_cmds=extra_cmds,
                output_callback=output_callback,
                save_inference_speed=save_inference_speed,
                expected_compared_events=expected_compared_events,
                qnn_intermediate_debugger=qnn_intermediate_debugger,
            )

    def get_qdq_module(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        is_conv_per_channel: Optional[bool] = True,
        is_linear_per_channel: Optional[bool] = False,
        custom_quant_annotations: Tuple[Callable] = (),
        quant_dtype: QuantDtype = QuantDtype.use_8a8w,
        dynamic_shapes: Dict = None,
        bypass_check: bool = False,
        block_size_map: Dict[str, Tuple] = None,
        submodule_qconfig_list: Optional[List[Tuple[Callable, ModuleQConfig]]] = None,
    ) -> torch.fx.GraphModule:
        m = torch.export.export(
            module, inputs, dynamic_shapes=dynamic_shapes, strict=True
        ).module()

        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            custom_annotations=custom_quant_annotations,
            per_channel_conv=is_conv_per_channel,
            per_channel_linear=is_linear_per_channel,
            submodule_qconfig_list=submodule_qconfig_list,
        )
        if block_size_map is not None:
            quantizer.set_block_size_map(block_size_map)
        prepared = prepare_pt2e(m, quantizer)

        prepared(*inputs)
        quantized_module = convert_pt2e(prepared)
        nodes = {node.target for node in quantized_module.graph.nodes}
        q_and_dq = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.torchao.quantize_affine.default,
            torch.ops.torchao.dequantize_affine.default,
        }
        if not bypass_check:
            self.assertTrue(nodes.intersection(q_and_dq))
        return quantized_module

    def get_prepared_qat_module(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        is_conv_per_channel: Optional[bool] = True,
        is_linear_per_channel: Optional[bool] = False,
        custom_quant_annotations: Tuple[Callable] = (),
        quant_dtype: QuantDtype = QuantDtype.use_8a8w,
        block_size_map: Dict[str, Tuple] = None,
        submodule_qconfig_list: Optional[List[Tuple[Callable, ModuleQConfig]]] = None,
    ) -> torch.fx.GraphModule:
        m = torch.export.export(module, inputs, strict=True).module()

        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            custom_annotations=custom_quant_annotations,
            per_channel_conv=is_conv_per_channel,
            per_channel_linear=is_linear_per_channel,
            is_qat=True,
            submodule_qconfig_list=submodule_qconfig_list,
        )
        if block_size_map is not None:
            quantizer.set_block_size_map(block_size_map)

        submodule_qconfig_list = submodule_qconfig_list or []
        quantizer.set_submodule_qconfig_list(submodule_qconfig_list)

        prepared = prepare_qat_pt2e(m, quantizer)
        return torchao.quantization.pt2e.move_exported_model_to_train(prepared)

    def get_converted_sgd_trained_module(
        self,
        ori_module: torch.nn.Module,
        prepared: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
    ) -> torch.fx.GraphModule:
        optimizer = torch.optim.SGD(prepared.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        output = prepared(*inputs)
        loss = criterion(output, ori_module(*inputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return convert_pt2e(prepared)

    def get_adb_tool(self, pte_fname):
        adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=self.build_folder,
            pte_path=pte_fname,
            workspace="/data/local/tmp/qnn_executorch_test",
            device_id=self.device,
            host_id=self.host,
            soc_model=self.model,
            error_only=self.error_only,
            target=self.target,
        )
        return adb

    def split_graph(self, division: int):
        class SplitGraph(ExportPass):
            """
            Split graph based on number of nodes.
            """

            def __init__(self, division):
                super().__init__()
                self.division = division

            def _is_legit_node(self, node):
                # skip dq_ops for frozen_params
                return node.op == "call_function" and node.target not in dq_ops

            def _insert_clone(
                self, graph_module: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                # Count the total of nodes in the graph
                num_graph_nodes = 0
                for node in graph_module.graph.nodes:
                    num_graph_nodes += 1 if node.op == "call_function" else 0

                # Compute how many nodes in one share
                shares = num_graph_nodes // self.division

                # Insert clone op to split model based on the shares
                num_graph_nodes = 0
                for node in graph_module.graph.nodes:
                    if not self._is_legit_node(node):
                        continue

                    num_graph_nodes += 1
                    if num_graph_nodes % shares != 0:
                        continue

                    with graph_module.graph.inserting_after(node):
                        users = list(node.users.keys())
                        inserted_node = graph_module.graph.create_node(
                            "call_function",
                            exir_ops.edge.dim_order_ops._clone_dim_order.default,
                            (node,),
                        )
                        inserted_node.meta["val"] = node.meta["val"]
                        if "quant_attrs" in node.meta:
                            inserted_node.meta["quant_attrs"] = node.meta["quant_attrs"]
                        for user in users:
                            user.replace_input_with(node, inserted_node)

            def call(self, graph_module: torch.fx.GraphModule):
                self._insert_clone(graph_module)
                graph_module.recompile()
                return PassResult(graph_module, True)

        return SplitGraph, {
            QCOM_PASS_ACTIVATE_KEY: True,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {"division": division},
        }
