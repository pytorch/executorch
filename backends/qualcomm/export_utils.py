# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors
import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torchao
from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
    QNNIntermediateDebugger,
)
from executorch.backends.qualcomm.quantizer.quantizer import (
    ModuleQConfig,
    QnnQuantizer,
    QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    LpaiHardwareVersion,
    QcomChipset,
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpPerformanceMode,
    QnnExecuTorchLpaiTargetEnv,
    QnnExecuTorchOpPackageOptions,
)
from executorch.backends.qualcomm.utils.constants import (
    HEXAGON_SDK_ROOT,
    HEXAGON_TOOLS_ROOT,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_lpai_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_qnn_context_binary_alignment,
    get_sdk_build_id,
    get_soc_to_htp_arch_map,
    get_soc_to_lpai_hw_ver_map,
    is_qnn_sdk_version_less_than,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization.pt2e import MovingAverageMinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)


@dataclass
class QnnConfig:
    """
    A configuration used as input to QNN ExecuTorch’s lowering API.
    This config initialization currently supports:
    1. Provide command-line arguments paired with setup_common_args_and_variables.
    2. Provide a json file that stores desired config.

    Attributes:
        backend (str): The target backend, such as htp, gpu, etc. QnnConfig will then parse this to type QnnExecuTorchBackendType.
        soc_model (QcomChipset): The target Qualcomm System on Chip (SoC) model.
        build_folder (str): Path to cmake binary directory for target platform, e.g., /path/to/build-android.
        direct_build_folder (str): Path to cmake binary directory for direct_mode. E.g., path/to/build-direct.
        target (str): Target platform for deployment.
        online_prepare (bool): Compose QNN graph on device if set to True.
        shared_buffer (bool): Enables usage of shared buffer(zero-copy mechanism) between application and backend for graph I/O during runtime.
        dump_intermediate_outputs (bool): Enables dumping model intermediate outputs.
        profile_level (int): Level of profiling in runtime.
        enable_x86_64: Enable x86_64 simulator execution.
        host (str): Hostname where android device is connected.
        device (str): Serial number for android device communicated via ADB.
        port (int): IPC port for delivering execution result
        ip (str): IPC address for delivering execution result.
        skip_delegate_node_ids (str): If specified, skip delegation for the specified node based on node ids. Node ids should be separated by comma. e.g., aten_relu_default_10,aten_relu_default_2
        skip_delegate_node_ops (str): If specified, skip delegation for the specified op. Node ops should be separated by comma. e.g., aten.add.Tensor,aten.relu.default
        compile_only (bool): If specified, only compile the model.
        pre_gen_pte (str): Run the pre-generated pte in the given directory.
        skip_push: If specified, skip pushing files to device. Assumes all required files are on device already.
        ci (bool): This flag is for Continuous Integration(CI) purpose and is NOT recommended to turn on for typical use cases. It will use random inputs instead of real inputs.
        seed (int): Set the seed for generating random numbers in both torch and random.
        htp_performance_mode (QnnExecuTorchHtpPerformanceMode, optional): Option to set the performance mode for htp backend.
    """

    soc_model: str
    build_folder: str
    direct_build_folder: Optional[str] = None
    backend: str = "htp"
    target: str = "aarch64-android"
    online_prepare: Optional[bool] = False
    shared_buffer: Optional[bool] = False
    dump_intermediate_outputs: Optional[bool] = False
    profile_level: Optional[int] = 0
    enable_x86_64: Optional[bool] = False
    host: Optional[str] = None
    device: Optional[str] = None
    port: Optional[str] = -1
    ip: Optional[str] = ""
    skip_delegate_node_ids: Optional[str] = None
    skip_delegate_node_ops: Optional[str] = None
    compile_only: Optional[bool] = False
    pre_gen_pte: Optional[str] = None
    skip_push: Optional[bool] = False
    ci: Optional[bool] = False
    seed: Optional[int] = None
    htp_performance_mode: QnnExecuTorchHtpPerformanceMode = (
        QnnExecuTorchHtpPerformanceMode.kHtpBurst,
    )

    def __post_init__(self):
        assert self.soc_model, "Please provide the soc_model"
        assert self.build_folder, "Please provide the build_folder."
        assert not (
            self.compile_only and self.pre_gen_pte
        ), "Cannot set both compile_only and pre_gen_pte as true"
        assert (
            "QNN_SDK_ROOT" in os.environ
        ), "Environment variable QNN_SDK_ROOT must be set."
        if (not self.compile_only and not self.enable_x86_64) and self.device is None:
            raise RuntimeError(
                "device serial is required if not compile only or run on x86 emulator. Please specify a device serial."
            )
        if self.backend == "lpai":
            if self.soc_model not in get_soc_to_lpai_hw_ver_map():
                raise RuntimeError(
                    f"Target soc_model({self.soc_model}) doesn't support LPAI backend. \n"
                    "Please choose the following SOC: "
                    f"{list(get_soc_to_lpai_hw_ver_map().keys())}"
                )
            elif get_soc_to_lpai_hw_ver_map()[
                self.soc_model
            ] == LpaiHardwareVersion.V6 and is_qnn_sdk_version_less_than("2.39"):
                raise RuntimeError(
                    f"Target soc_model({self.soc_model}) with LPAI backend v6 requires QNN SDK version >= 2.39. \n"
                    f"Current QNN SDK version: {get_sdk_build_id()}"
                )

        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.backend = get_backend_type(self.backend)
        self.skip_delegate_node_ids, self.skip_delegate_node_ops = (
            self._parse_skip_delegation_node(
                self.skip_delegate_node_ids, self.skip_delegate_node_ops
            )
        )

    @classmethod
    def load_config(cls, config: Union[argparse.Namespace, str]) -> "QnnConfig":
        """
        config (Union[argparse.Namespace, str]): Accepts either a parser generated from setup_common_args_and_variables() or a json file.
        """
        qnn_config = None
        if isinstance(config, argparse.Namespace):
            logging.info("Using parser's config")
            args_dict = vars(config)
            matched_keys = {f.name for f in fields(QnnConfig)}
            config = {k: v for k, v in args_dict.items() if k in matched_keys}
            qnn_config = cls(**config)
        elif isinstance(config, str):
            logging.info(f"Using {config}'s config.")

            with open(config) as f:
                qnn_config = cls(**json.load(f))
        else:
            raise TypeError(
                f"Invalid config type {type(config).__name__}. Expected argparse.Namespace or str."
            )

        return qnn_config

    def _parse_skip_delegation_node(
        self, skip_delegate_node_ids, skip_delegate_node_ops
    ):
        skip_node_id_set = set()
        skip_node_op_set = set()

        if skip_delegate_node_ids:
            skip_node_id_set = set(map(str, skip_delegate_node_ids.split(",")))
            print("Skipping following node ids: ", skip_node_id_set)

        if skip_delegate_node_ops:
            skip_node_op_set = set(map(str, skip_delegate_node_ops.split(",")))
            print("Skipping following node ops: ", skip_node_op_set)

        return skip_node_id_set, skip_node_op_set


class SimpleADB:
    """
    A wrapper class for communicating with Android device

    Attributes:
        qnn_config: (QnnConfig): A config class that saves qnn lowering and execution configuration.
        pte_path (Union[str, list]): Path where executorch binary was stored. If there are multiple pte files, provide a list of pte paths.
        workspace (str): Folder for storing artifacts on android device
        error_only (bool): Redirect stdio and leave error messages only
        runner (str): Runtime executor binary
        expected_input_shape (Tuple[torch.Size]): Input shape of dynamic graph
        expected_output_shape (Tuple[torch.Size]): Output shape of dynamic graph
    """

    def __init__(
        self,
        qnn_config: QnnConfig,
        pte_path: Union[str, list],
        workspace,
        error_only=False,
        runner=None,
        expected_input_shape=None,
        expected_output_shape=None,
    ):
        if runner is None:
            runner = (
                "examples/qualcomm/executor_runner/qnn_executor_runner"
                if qnn_config.direct_build_folder is None
                else "examples/qualcomm/direct_executor_runner/qnn_executor_direct_runner"
            )
        self.runner = runner
        if qnn_config.direct_build_folder:
            required_env = [HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT]
            assert all(
                var in os.environ for var in required_env
            ), f"Please ensure the following environment variables are set: {required_env}"
            self.hexagon_sdk_root = os.getenv(HEXAGON_SDK_ROOT)
            self.hexagon_tools_root = os.getenv(HEXAGON_TOOLS_ROOT)
            logging.info(f"{HEXAGON_SDK_ROOT}={self.hexagon_sdk_root}")
            logging.info(f"{HEXAGON_TOOLS_ROOT}={self.hexagon_tools_root}")
        self.qnn_config = qnn_config
        self.qnn_sdk = os.getenv("QNN_SDK_ROOT")
        self.build_path = qnn_config.build_folder
        self.direct_build_folder = qnn_config.direct_build_folder
        self.pte_path = pte_path if isinstance(pte_path, list) else [pte_path]
        if qnn_config.pre_gen_pte:
            self.pte_path = [
                os.path.join(qnn_config.pre_gen_pte, os.path.basename(p))
                for p in self.pte_path
            ]
            assert all(
                os.path.exists(p) for p in self.pte_path
            ), f"{self.pte_path} not found. Please ensure there are pregenerated pte files under pre_gen_pte path."
            logging.info(
                f"Pregenerated pte path given. Using pre_gen_pte path: {self.pte_path}"
            )
        self.workspace = workspace
        self.device_id = qnn_config.device
        self.host_id = qnn_config.host
        if len(self.pte_path) > 0:
            self.working_dir = Path(self.pte_path[0]).parent.absolute()
        else:
            self.working_dir = Path.cwd()
        self.input_list_filename = "input_list.txt"
        self.etdump_path = f"{self.workspace}/etdump.etdp"
        self.dump_intermediate_outputs = qnn_config.dump_intermediate_outputs
        self.debug_output_path = f"{self.workspace}/debug_output.bin"
        self.output_folder = f"{self.workspace}/outputs"
        self.htp_arch = get_soc_to_htp_arch_map()[qnn_config.soc_model]
        self.lpai_hw_ver = get_soc_to_lpai_hw_ver_map().get(qnn_config.soc_model, None)
        self.error_only = error_only
        self.shared_buffer = qnn_config.shared_buffer
        self.target = qnn_config.target
        self.expected_input_shape = expected_input_shape
        self.expected_output_shape = expected_output_shape
        self.extra_cmds = ""
        self.skip_push = qnn_config.skip_push
        self.backend_library_paths = {}

        if self.direct_build_folder:
            direct_general_artifacts = [
                f"{self.build_path}/examples/qualcomm/direct_executor_runner/libqnn_executorch_stub.so",
            ]
            self.backend_library_paths.update(
                {
                    QnnExecuTorchBackendType.kHtpBackend: [
                        f"{self.direct_build_folder}/backends/qualcomm/libqnn_executorch_backend.so",
                        f"{self.direct_build_folder}/backends/qualcomm/qnn_executorch/direct_mode/libqnn_executorch_skel.so",
                        f"{self.qnn_sdk}/lib/hexagon-v{self.htp_arch}/unsigned/libQnnHtpV{self.htp_arch}.so",
                        f"{self.qnn_sdk}/lib/hexagon-v{self.htp_arch}/unsigned/libQnnSystem.so",
                        f"{self.hexagon_tools_root}/Tools/target/hexagon/lib/v{self.htp_arch}/G0/pic/libc++abi.so.1",
                        f"{self.hexagon_tools_root}/Tools/target/hexagon/lib/v{self.htp_arch}/G0/pic/libc++.so.1",
                    ],
                    QnnExecuTorchBackendType.kLpaiBackend: [
                        f"{self.qnn_sdk}/lib/lpai-v{self.lpai_hw_ver}/signed/libqnn_executorch_backend.so",
                        f"{self.qnn_sdk}/lib/lpai-v{self.lpai_hw_ver}/signed/libqnn_executorch_skel.so",
                        f"{self.qnn_sdk}/lib/lpai-v{self.lpai_hw_ver}/signed/libQnnLpai.so",
                        f"{self.qnn_sdk}/lib/lpai-v{self.lpai_hw_ver}/signed/libQnnSystem.so",
                        f"{self.qnn_sdk}/lib/lpai-v{self.lpai_hw_ver}/signed/libc++abi.so.1",
                        f"{self.qnn_sdk}/lib/lpai-v{self.lpai_hw_ver}/signed/libc++.so.1",
                    ],
                }
            )
            for _, library_paths in self.backend_library_paths.items():
                library_paths.extend(direct_general_artifacts)
        else:
            traditional_general_artifacts = [
                f"{self.qnn_sdk}/lib/{self.target}/libQnnSystem.so",
                f"{self.build_path}/backends/qualcomm/libqnn_executorch_backend.so",
                f"{self.qnn_sdk}/lib/{self.target}/libQnnModelDlc.so",
            ]
            self.backend_library_paths.update(
                {
                    QnnExecuTorchBackendType.kHtpBackend: [
                        f"{self.qnn_sdk}/lib/{self.target}/libQnnHtp.so",
                        (
                            f"{self.qnn_sdk}/lib/hexagon-v{self.htp_arch}/"
                            f"unsigned/libQnnHtpV{self.htp_arch}Skel.so"
                        ),
                        (
                            f"{self.qnn_sdk}/lib/{self.target}/"
                            f"libQnnHtpV{self.htp_arch}Stub.so"
                        ),
                        f"{self.qnn_sdk}/lib/{self.target}/libQnnHtpPrepare.so",
                    ],
                    QnnExecuTorchBackendType.kGpuBackend: [
                        f"{self.qnn_sdk}/lib/{self.target}/libQnnGpu.so",
                    ],
                    # please note that users need to sign LPAI related libs manually
                    QnnExecuTorchBackendType.kLpaiBackend: [
                        f"{self.qnn_sdk}/lib/{self.target}/libQnnLpai.so",
                        (
                            f"{self.qnn_sdk}/lib/lpai-v{self.lpai_hw_ver}/"
                            f"signed/libQnnLpaiSkel.so"
                        ),
                        f"{self.qnn_sdk}/lib/{self.target}/libQnnLpaiStub.so",
                    ],
                }
            )
            for _, library_paths in self.backend_library_paths.items():
                library_paths.extend(traditional_general_artifacts)

    def _adb(self, cmd, output_callback: Optional[Callable[[str], None]] = None):
        if not self.host_id:
            cmds = ["adb", "-s", self.device_id]
        else:
            cmds = ["adb", "-H", self.host_id, "-s", self.device_id]
        cmds.extend(cmd)

        if output_callback:
            result = subprocess.run(
                cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            output_callback(result)
        else:
            result = subprocess.run(
                cmds, stdout=subprocess.DEVNULL if self.error_only else sys.stdout
            )
        if result.returncode != 0:
            raise RuntimeError(f"adb command failed: {cmds}")

    def push(  # noqa: C901
        self,
        inputs=None,
        files=None,
        backends: Optional[Set[QnnExecuTorchBackendType]] = None,
        init_env=True,
    ):
        # Assume all required files are on device already
        if self.skip_push:
            return

        artifacts = [*self.pte_path, f"{self.build_path}/{self.runner}"]
        if init_env:
            self._adb(["shell", f"rm -rf {self.workspace}"])
            self._adb(["shell", f"mkdir -p {self.workspace}"])

            if backends is None:
                backends = {self.qnn_config.backend}

            # backend libraries
            for backend in backends:
                artifacts.extend(self.backend_library_paths[backend])

            # Ensure that all necessary library artifacts exists.
            missing = [path for path in artifacts if not os.path.exists(path)]
            assert not missing, "Missing the following libraries:\n" + "\n".join(
                f"  {p}" for p in missing
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_list_file, input_files = generate_inputs(
                tmp_dir, self.input_list_filename, inputs
            )

            if input_list_file is not None:
                # prepare input list
                artifacts.append(input_list_file)

            for artifact in artifacts:
                self._adb(["push", artifact, self.workspace])

            # input data
            for file_name in input_files:
                self._adb(["push", file_name, self.workspace])

            # dynamic shape related
            if self.expected_input_shape and self.expected_output_shape:
                shape_info = {
                    "input_shape": self.expected_input_shape,
                    "output_shape": self.expected_output_shape,
                }
                for name, shapes in shape_info.items():
                    with open(f"{tmp_dir}/{name}.txt", "w") as f:
                        for s in shapes:
                            f.write(str(tuple(s)).strip("()") + "\n")
                    self._adb(["push", f"{tmp_dir}/{name}.txt", self.workspace])
                    self.extra_cmds += f" --{name}_path {name}.txt"

        # custom files
        if files is not None:
            for file_name in files:
                self._adb(["push", file_name, self.workspace])

    def execute(
        self,
        custom_runner_cmd=None,
        method_index=0,
        output_callback: Optional[Callable[[str], None]] = None,
        iteration=1,
    ):
        self._adb(["shell", f"mkdir -p {self.output_folder}"])
        # run the delegation
        if custom_runner_cmd is None:
            qnn_executor_runner_args = (
                " ".join(
                    [
                        f"--model_path {os.path.basename(self.pte_path[0])}",
                        f"--output_folder_path {self.output_folder}",
                        f"--input_list_path {self.input_list_filename}",
                        f"--etdump_path {self.etdump_path}",
                        "--shared_buffer" if self.shared_buffer else "",
                        f"--debug_output_path {self.debug_output_path}",
                        (
                            "--dump_intermediate_outputs"
                            if self.dump_intermediate_outputs
                            else ""
                        ),
                        f"--method_index {method_index}",
                        "" if self.direct_build_folder else f"--iteration {iteration}",
                    ]
                )
                + self.extra_cmds
            )
            if self.qnn_config.direct_build_folder:
                qnn_executor_runner_args = " ".join(
                    [
                        qnn_executor_runner_args,
                        f"--domain_id {get_dsp_id(self.qnn_config.backend)}",
                    ]
                )
            qnn_executor_runner_cmds = " ".join(
                [
                    f"cd {self.workspace} &&",
                    f"chmod +x {os.path.basename(self.runner)} &&",
                    f"export LD_LIBRARY_PATH=. && export ADSP_LIBRARY_PATH=. && echo 0x0C > {os.path.basename(self.runner)}.farf && ./{os.path.basename(self.runner)} {qnn_executor_runner_args}",
                ]
            )
        else:
            qnn_executor_runner_cmds = custom_runner_cmd
        self._adb(
            ["shell", f"{qnn_executor_runner_cmds}"], output_callback=output_callback
        )

    def pull(self, host_output_path, device_output_path=None, callback=None):
        if device_output_path is None:
            device_output_path = self.output_folder
        self._adb(["pull", "-a", device_output_path, host_output_path])
        if callback:
            callback()

    def pull_etdump(self, output_path, callback=None):
        self._adb(["pull", self.etdump_path, output_path])
        if callback:
            callback()

    def pull_debug_output(self, etdump_path, debug_ouput_path, callback=None):
        self._adb(["pull", self.etdump_path, etdump_path])
        self._adb(["pull", self.debug_output_path, debug_ouput_path])
        if callback:
            callback()


def build_executorch_binary(
    model: torch.nn.Module,  # noqa: B006
    qnn_config: QnnConfig,
    file_name: str,
    dataset: List[torch.Tensor] | Callable[[torch.fx.GraphModule], None],
    quant_dtype: Optional[QuantDtype] = None,
    custom_quantizer: Optional[QnnQuantizer] = None,
    metadata=None,
    qnn_intermediate_debugger: QNNIntermediateDebugger = None,
    passes_job=None,
    passes_dependency=None,
    qat_training_data=None,
    op_package_options: QnnExecuTorchOpPackageOptions = None,
):
    """
    A function to generate an ExecuTorch binary for Qualcomm platforms.

    Attributes:
        model (torch.nn.Module): The model to be converted into an ExecuTorch binary.
        qnn_config: (QnnConfig): A config class that saves qnn lowering and execution configuration.
        file_name (str): Name for the output binary file (.pte).
        dataset (List[torch.Tensor] | Callable): A dataset for quantization calibration.
        quant_dtype (QuantDtype, optional): Data type for quantization.
        custom_quantizer (Callable, optional): Custom quantizer.
        metadata (dict, optional): An optional dictionary that maps each method name to a constant value in eager mode.
        passes_job (OrderedDict, optional): Custom passes job in to_edge_transform_and_lower, users can enable/disable specific passes or modify their attributes.
        passes_dependency (Dict, optional): A dictionary mapping each pass to its corresponding list of dependencies.
        qat_training_data (List[torch.Tensor], optional): A dataset for quantization aware training(QAT). Typically is a pair of tensors, such as [features, ground truth].
        op_package_options: Optional structure to specify op packages
            loaded and used by the backend.

    Returns:
        None: The function writes the output to a specified .pte file.
    """
    if qnn_config.pre_gen_pte:
        logging.info(
            f"Skip build_executorch_binary, using {file_name} under {qnn_config.pre_gen_pte}."
        )
        return

    sample_input = dataset[0]
    if (
        qnn_config.backend == QnnExecuTorchBackendType.kGpuBackend
        and not qnn_config.online_prepare
    ):
        raise RuntimeError(
            "Currently GPU backend only supports online_prepare. Please add --online_prepare flag."
        )
    if (
        qnn_config.backend == QnnExecuTorchBackendType.kLpaiBackend
        and qnn_config.online_prepare
    ):
        raise RuntimeError("Currently LPAI backend only supports offline_prepare.")
    backend_options = {
        QnnExecuTorchBackendType.kLpaiBackend: generate_lpai_compiler_spec(
            target_env=get_lpai_target_env(qnn_config)
        ),
        QnnExecuTorchBackendType.kGpuBackend: generate_gpu_compiler_spec(),
        QnnExecuTorchBackendType.kHtpBackend: generate_htp_compiler_spec(
            use_fp16=False if quant_dtype is not None else True,
            htp_performance_mode=qnn_config.htp_performance_mode,
        ),
    }[qnn_config.backend]
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, qnn_config.soc_model),
        backend_options=backend_options,
        online_prepare=qnn_config.online_prepare,
        profile_level=qnn_config.profile_level,
        shared_buffer=qnn_config.shared_buffer,
        dump_intermediate_outputs=qnn_config.dump_intermediate_outputs,
        op_package_options=op_package_options,
    )
    if quant_dtype is not None or custom_quantizer is not None:
        captured_model = torch.export.export(model, sample_input, strict=False).module()
        if qat_training_data:
            quantizer = custom_quantizer or make_quantizer(
                quant_dtype=quant_dtype,
                is_qat=True,
                backend=qnn_config.backend,
                soc_model=qnn_config.soc_model,
            )
            # qat training
            annotated_model = _qat_train(
                model, captured_model, quantizer, qat_training_data
            )
        else:
            quantizer = custom_quantizer or make_quantizer(
                quant_dtype=quant_dtype,
                backend=qnn_config.backend,
                soc_model=qnn_config.soc_model,
            )
            # ptq calibration
            with torch.no_grad():
                annotated_model = _ptq_calibrate(captured_model, quantizer, dataset)

        quantized_model = convert_pt2e(annotated_model)
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            quantized_model,
            sample_input,
            compile_spec,
            constant_methods=metadata,
            passes_job=passes_job,
            dep_table=passes_dependency,
            skip_node_id_set=qnn_config.skip_delegate_node_ids,
            skip_node_op_set=qnn_config.skip_delegate_node_ops,
            generate_etrecord=qnn_intermediate_debugger is not None,
        )
    else:
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            model,
            sample_input,
            compile_spec,
            constant_methods=metadata,
            passes_job=passes_job,
            skip_node_id_set=qnn_config.skip_delegate_node_ids,
            skip_node_op_set=qnn_config.skip_delegate_node_ops,
            generate_etrecord=qnn_intermediate_debugger is not None,
        )

    allocate_io = not (qnn_config.shared_buffer or qnn_config.direct_build_folder)
    executorch_config = ExecutorchBackendConfig(
        # For shared buffer, user must pass the memory address
        # which is allocated by RPC memory to executor runner.
        # Therefore, won't want to pre-allocate
        # by memory manager in runtime.
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=allocate_io,
            alloc_graph_output=allocate_io,
        ),
        segment_alignment=get_qnn_context_binary_alignment(),
    )
    pte_name = f"{file_name}.pte"
    exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
    with open(pte_name, "wb") as file:
        exec_prog_mgr.write_to_file(file)

    if qnn_intermediate_debugger:
        etrecord = exec_prog_mgr.get_etrecord()
        etrecord.update_representative_inputs(qnn_intermediate_debugger.sample_input)
        edge_ep = etrecord.graph_map[qnn_intermediate_debugger.reference_graph_name]
        # Use this edge_ep since edge_ep after etrecord serialize/deserialize will lose quant_attrs info.
        qnn_intermediate_debugger.set_edge_ep(edge_ep=edge_ep)
        etrecord_file_path = f"{os.path.dirname(pte_name)}/debug.etrecord"
        qnn_intermediate_debugger.set_etrecord_file_path(etrecord_file_path)
        etrecord.save(etrecord_file_path)

    if qnn_config.compile_only:
        sys.exit(0)


def make_quantizer(
    quant_dtype: Optional[QuantDtype] = QuantDtype.use_8a8w,
    custom_annotations=(),
    per_channel_conv=True,
    per_channel_linear=False,
    per_channel_embedding=False,
    act_observer=MovingAverageMinMaxObserver,
    act_symmetric=False,
    is_qat=False,
    submodule_qconfig_list: Optional[List[Tuple[Callable, ModuleQConfig]]] = None,
    backend=QnnExecuTorchBackendType.kHtpBackend,
    soc_model="SM8750",
    eps=None,
):
    quantizer = QnnQuantizer(backend=backend, soc_model=getattr(QcomChipset, soc_model))
    quantizer.add_custom_quant_annotations(custom_annotations)
    quantizer.set_default_quant_config(
        quant_dtype,
        is_qat=is_qat,
        is_conv_per_channel=per_channel_conv,
        is_linear_per_channel=per_channel_linear,
        is_embedding_per_channel=per_channel_embedding,
        act_observer=act_observer,
        act_symmetric=act_symmetric,
        eps=eps,
    )
    submodule_qconfig_list = submodule_qconfig_list or []
    quantizer.set_submodule_qconfig_list(submodule_qconfig_list)
    return quantizer


def get_lpai_target_env(qnn_config: QnnConfig):
    if qnn_config.enable_x86_64:
        return QnnExecuTorchLpaiTargetEnv.kX86
    elif qnn_config.direct_build_folder:
        return QnnExecuTorchLpaiTargetEnv.kAdsp
    return QnnExecuTorchLpaiTargetEnv.kArm


def get_backend_type(backend: str):
    return getattr(QnnExecuTorchBackendType, f"k{backend.title()}Backend")


def get_dsp_id(backend):
    dsp_id_map = {
        QnnExecuTorchBackendType.kLpaiBackend: 0,
        QnnExecuTorchBackendType.kHtpBackend: 3,
    }
    if backend not in dsp_id_map:
        raise ValueError(
            f"Unsupported backend {backend} for direct mode. "
            f"Supported: {list(dsp_id_map.keys())}"
        )
    return dsp_id_map[backend]


def setup_common_args_and_variables():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        help="To reduce the effort of providing a lot of command-line arguments, users can choose to save all arguments to a .json file and pass it in. Please refer to executorch/examples/qualcomm/executor_runner/sample_config.json for sample.",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-m",
        "--soc_model",
        "--model",  # Deprecate this flag in future.
        help="SoC model of current device. e.g. 'SM8550' for Snapdragon 8 Gen 2.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-b",
        "--build_folder",
        help="path to cmake binary directory for target platform, e.g., /path/to/build-android",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-H",
        "--host",
        help="hostname where android device is connected.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--online_prepare",
        help="If specified, compose QNN graph on device.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--ip",
        help="IPC address for delivering execution result",
        default="",
        type=str,
    )

    parser.add_argument(
        "--port",
        help="IPC port for delivering execution result",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "-S",
        "--skip_delegate_node_ids",
        help="If specified, skip delegation for the specified node based on node ids. Node ids should be separated by comma. e.g., aten_relu_default_10,aten_relu_default_2",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-f",
        "--skip_delegate_node_ops",
        help="If specified, skip delegation for the specified op. Node ops should be separated by comma. e.g., aten.add.Tensor,aten.relu.default",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-c",
        "--compile_only",
        help="If specified, only compile the model.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-s",
        "--device",
        help="serial number for android device communicated via ADB.",
        type=str,
    )

    parser.add_argument(
        "--backend",
        help="Backend to be deployed ('htp'/'gpu'/'lpai' are currently supported).",
        choices=["htp", "gpu", "lpai"],
        default="htp",
        type=str,
    )

    parser.add_argument(
        "-z",
        "--shared_buffer",
        help="Enables usage of shared buffer(zero-copy mechanism) between application and backend for graph I/O.",
        action="store_true",
    )

    parser.add_argument(
        "--skip_push",
        help="If specified, skip pushing files to device.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-D",
        "--dump_intermediate_outputs",
        help="If specified, enable dump intermediate outputs",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--profile_level",
        type=int,
        help="Profiling level of the delegate and QNN backend. 0=Off, 1=Basic(Currently not supported), 2=Detailed, 3=Optrace.",
        choices=[0, 2, 3],
        default=0,
    )

    parser.add_argument(
        "-x",
        "--enable_x86_64",
        help="Enable unittest to be executed on x86_64 platform",
        action="store_true",
    )

    parser.add_argument(
        "--ci",
        help="This flag is for Continuous Integration(CI) purpose and is NOT recommended to turn on for typical use cases. It will use random inputs instead of real inputs.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--seed",
        help="Set the seed for generating random numbers in both torch and random.",
        type=int,
    )

    parser.add_argument(
        "-t",
        "--target",
        help="Target platform for deployment",
        choices=[
            "aarch64-android",
            "aarch64-oe-linux-gcc9.3",
            "aarch64-oe-linux-gcc11.2",
        ],
        default="aarch64-android",
        type=str,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the pre-generated pte in the given directory.",
        type=str,
    )

    parser.add_argument(
        "--direct_build_folder",
        help="Path to cmake binary directory for direct_mode. E.g., path/to/build-direct."
        "If enabled, run self-defined protocol to control fastrpc communication.",
        type=str,
    )

    parser.add_argument(
        "--htp_performance_mode",
        type=int,
        choices=list(QnnExecuTorchHtpPerformanceMode),
        help="Specify performance mode for htp from 0-8, default to burst(2). For more info, refer to qc_schema.py",
        default=2,
    )

    return parser


def generate_inputs(
    dest_path: str,
    input_list_filename: str,
    inputs=None,
    prefix_input_filename: str = "",
):

    input_list_file = None
    input_files = []

    def prepare_input_file(tensor, fd, index, sub_index):
        # transform torch.Tensor to raw file
        input_file_name = f"{prefix_input_filename}_input_{index}_{sub_index}.raw"
        input_file_path = f"{dest_path}/{input_file_name}"
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        tensor.detach().numpy().tofile(input_file_path)
        input_files.append(input_file_path)
        # prepare input_list
        if sub_index > 0:
            fd.write(" ")
        fd.write(input_file_name)

    # Prepare input data
    if inputs is not None:
        input_list_file = f"{dest_path}/{input_list_filename}"

        with open(input_list_file, "w") as f:
            for idx, data in enumerate(inputs):
                sub_index = 0
                for d in data:
                    if isinstance(d, (list, tuple)):
                        for sub_d in d:
                            prepare_input_file(sub_d, f, idx, sub_index)
                            sub_index += 1
                    else:
                        prepare_input_file(d, f, idx, sub_index)
                        sub_index += 1

                f.write("\n")

    return input_list_file, input_files


def _qat_train(ori_model, captured_model, quantizer, dataset):
    data, targets = dataset
    annotated_model = torchao.quantization.pt2e.move_exported_model_to_train(
        prepare_qat_pt2e(captured_model, quantizer)
    )
    optimizer = torch.optim.SGD(annotated_model.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    for i, d in enumerate(data):
        print(f"Epoch {i}")
        if i > 3:
            # Freeze quantizer parameters
            annotated_model.apply(
                torchao.quantization.pt2e.fake_quantize.disable_observer
            )
        if i > 2:
            # Freeze batch norm mean and variance estimates
            annotated_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        output = annotated_model(*d)
        loss = criterion(output, targets[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return convert_pt2e(
        torchao.quantization.pt2e.move_exported_model_to_eval(annotated_model),
    )


def _ptq_calibrate(captured_model, quantizer, dataset):
    annotated_model = prepare_pt2e(captured_model, quantizer)
    print("Quantizing(PTQ) the model...")
    # calibration
    if callable(dataset):
        dataset(annotated_model)
    else:
        for data in dataset:
            annotated_model(*data)
    return annotated_model
