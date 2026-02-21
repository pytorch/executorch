# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of showcasing registering custom operator through torch library API."""
import json
import os
import subprocess
import sys
from multiprocessing.connection import Client

import numpy as np
import torch

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    HtpArch,
    QcomChipset,
    QnnExecuTorchOpPackageInfo,
    QnnExecuTorchOpPackageOptions,
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    generate_inputs,
    make_output_dir,
    make_quantizer,
    setup_common_args_and_variables,
    SimpleADB,
)
from torch.library import impl, Library

my_op_lib = Library("my_ops", "DEF")

# registering an operator that multiplies input tensor by 3 and returns it.
my_op_lib.define("mul3(Tensor input) -> Tensor")  # should print 'mul3'


@impl(my_op_lib, "mul3", dispatch_key="CompositeExplicitAutograd")
def mul3_impl(a: torch.Tensor) -> torch.Tensor:
    return a * 3


# registering the out variant.
my_op_lib.define(
    "mul3.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)"
)  # should print 'mul3.out'


@impl(my_op_lib, "mul3.out", dispatch_key="CompositeExplicitAutograd")
def mul3_out_impl(a: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(a)
    out.mul_(3)
    return out


# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul3.default(a)


def annotate_custom(gm: torch.fx.GraphModule) -> None:
    """
    This function is specific for custom op.
    The source_fn of the rewritten nn module turns out to be "my_ops.mul3.default"
    """
    from executorch.backends.qualcomm.quantizer.annotators import _is_annotated

    from executorch.backends.qualcomm.quantizer.qconfig import (
        get_ptq_per_channel_quant_config,
    )
    from torch.fx import Node
    from torchao.quantization.pt2e.quantizer import QuantizationAnnotation
    from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

    quantization_config = get_ptq_per_channel_quant_config()
    for node in gm.graph.nodes:
        if node.target != torch.ops.my_ops.mul3.default:
            continue

        # skip annotation if it is already annotated
        if _is_annotated([node]):
            continue

        input_qspec_map = {}
        input_act = node.args[0]
        assert isinstance(input_act, Node)
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


def _run(cmd, cwd=None):
    subprocess.run(cmd, stdout=sys.stdout, cwd=cwd, check=True)


def prepare_op_package(
    workspace: str, op_package_dir: str, arch: HtpArch, build_op_package: bool
):
    if build_op_package:
        _run(["rm", "-rf", "build"], cwd=op_package_dir)
        _run(["make", "htp_x86", "htp_aarch64", f"htp_v{arch}"], cwd=op_package_dir)
        _run(
            [
                "cp",
                f"{op_package_dir}/build/hexagon-v{arch}/libQnnExampleOpPackage.so",
                f"{op_package_dir}/build/hexagon-v{arch}/libQnnExampleOpPackage_HTP.so",
            ]
        )

    op_package_paths = [
        f"{op_package_dir}/build/hexagon-v{arch}/libQnnExampleOpPackage_HTP.so",
        f"{op_package_dir}/build/aarch64-android/libQnnExampleOpPackage.so",
    ]

    op_package_infos_HTP = QnnExecuTorchOpPackageInfo()
    op_package_infos_HTP.interface_provider = "ExampleOpPackageInterfaceProvider"
    op_package_infos_HTP.op_package_name = "ExampleOpPackage"
    op_package_infos_HTP.op_package_path = f"{workspace}/libQnnExampleOpPackage_HTP.so"
    op_package_infos_HTP.target = QnnExecuTorchOpPackageTarget.HTP
    op_package_infos_HTP.custom_op_name = "my_ops.mul3.default"
    op_package_infos_HTP.qnn_op_type_name = "ExampleCustomOp"
    op_package_infos_HTP.platform = QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID
    op_package_infos_aarch64_CPU = QnnExecuTorchOpPackageInfo()
    op_package_infos_aarch64_CPU.interface_provider = (
        "ExampleOpPackageInterfaceProvider"
    )
    op_package_infos_aarch64_CPU.op_package_name = "ExampleOpPackage"
    op_package_infos_aarch64_CPU.op_package_path = (
        f"{workspace}/libQnnExampleOpPackage.so"
    )
    op_package_infos_aarch64_CPU.target = QnnExecuTorchOpPackageTarget.CPU
    op_package_infos_aarch64_CPU.custom_op_name = "my_ops.mul3.default"
    op_package_infos_aarch64_CPU.qnn_op_type_name = "ExampleCustomOp"
    op_package_infos_aarch64_CPU.platform = (
        QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID
    )
    op_package_infos_x86_CPU = QnnExecuTorchOpPackageInfo()
    op_package_infos_x86_CPU.interface_provider = "ExampleOpPackageInterfaceProvider"
    op_package_infos_x86_CPU.op_package_name = "ExampleOpPackage"
    op_package_infos_x86_CPU.op_package_path = (
        f"{op_package_dir}/build/x86_64-linux-clang/libQnnExampleOpPackage.so"
    )
    op_package_infos_x86_CPU.target = QnnExecuTorchOpPackageTarget.CPU
    op_package_infos_x86_CPU.custom_op_name = "my_ops.mul3.default"
    op_package_infos_x86_CPU.qnn_op_type_name = "ExampleCustomOp"
    op_package_infos_x86_CPU.platform = QnnExecuTorchOpPackagePlatform.X86_64
    op_package_options = QnnExecuTorchOpPackageOptions()
    op_package_options.op_package_infos = [
        op_package_infos_x86_CPU,
        op_package_infos_aarch64_CPU,
        op_package_infos_HTP,
    ]

    return op_package_options, op_package_paths


def main(args):
    if args.build_op_package:
        if "HEXAGON_SDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable HEXAGON_SDK_ROOT must be set")
        print(f"HEXAGON_SDK_ROOT={os.getenv('HEXAGON_SDK_ROOT')}")

        if "ANDROID_NDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable ANDROID_NDK_ROOT must be set")
        print(f"ANDROID_NDK_ROOT={os.getenv('ANDROID_NDK_ROOT')}")

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    quant_dtype = QuantDtype.use_8a8w
    if args.use_fp16:
        quant_dtype = None

    instance = Model()
    pte_filename = "custom_qnn"
    sample_input = (torch.ones(1, 32, 28, 28),)
    workspace = f"/data/local/tmp/executorch/{pte_filename}"

    soc_info = _soc_info_table[getattr(QcomChipset, args.model)]

    op_package_options, op_package_paths = prepare_op_package(
        workspace,
        args.op_package_dir,
        soc_info.htp_info.htp_arch,
        args.build_op_package,
    )
    quantizer = make_quantizer(
        quant_dtype=quant_dtype, custom_annotations=(annotate_custom,)
    )

    build_executorch_binary(
        instance,
        sample_input,
        args.model,
        f"{args.artifact}/{pte_filename}",
        sample_input,
        op_package_options=op_package_options,
        quant_dtype=quant_dtype,
        custom_quantizer=quantizer,
    )

    if args.compile_only:
        sys.exit(0)

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    if args.enable_x86_64:
        input_list_filename = "input_list.txt"
        generate_inputs(args.artifact, input_list_filename, sample_input)
        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
        target = "x86_64-linux-clang"

        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                f"./{args.build_folder}/examples/qualcomm/executor_runner/qnn_executor_runner",
                f"--model_path {args.artifact}/{pte_filename}.pte",
                f"--input_list_path {args.artifact}/{input_list_filename}",
                f"--output_folder_path {output_data_folder}",
            ]
        )
        subprocess.run(
            runner_cmd,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )
    else:
        # setup required paths accordingly
        # qnn_sdk       : QNN SDK path setup in environment variable
        # artifact_path : path where artifacts were built
        # pte_path      : path where executorch binary was stored
        # device_id     : serial number of android device
        # workspace     : folder for storing artifacts on android device
        adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=f"{args.build_folder}",
            pte_path=f"{args.artifact}/{pte_filename}.pte",
            workspace=workspace,
            device_id=args.device,
            host_id=args.host,
            soc_model=args.model,
            shared_buffer=args.shared_buffer,
            target=args.target,
        )
        adb.push(inputs=sample_input, files=op_package_paths)
        adb.execute()
        adb.pull(host_output_path=args.artifact)

    x86_golden = instance(*sample_input)
    device_output = torch.from_numpy(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
    ).reshape(x86_golden.size())
    result = torch.all(torch.isclose(x86_golden, device_output, atol=1e-2)).tolist()

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "is_close": result,
                    }
                )
            )
    else:
        print(f"is_close? {result}")
        if not result:
            print(f"x86_golden {x86_golden}")
            print(f"device_out {device_output}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./custom_op",
        default="./custom_op",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--op_package_dir",
        help="Path to operator package which generates from QNN.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-F",
        "--use_fp16",
        help="If specified, will run in fp16 precision and discard ptq setting",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--build_op_package",
        help="Build op package based on op_package_dir. Please set up "
        "`HEXAGON_SDK_ROOT` and `ANDROID_NDK_ROOT` environment variable. "
        "And add clang compiler into `PATH`. Please refer to  Qualcomm AI Engine "
        "Direct SDK document to get more details",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    args.validate(args)

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
