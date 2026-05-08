# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of registering single output custom operator through torch library API."""

import json
import os
import subprocess
import sys
from multiprocessing.connection import Client

import numpy as np
import torch

from executorch.backends.qualcomm.custom_op.annotator import (
    CustomOpsQuantAnnotator,
    IOQuantConfig,
)
from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    generate_inputs,
    get_backend_type,
    make_quantizer,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    get_ptq_per_channel_quant_config,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    QcomChipset,
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)
from executorch.examples.qualcomm.utils import make_output_dir
from torch.library import impl, Library

my_op_lib = Library("my_ops", "DEF")

# registering an operator that multiplies input tensor by 3 and returns it.
my_op_lib.define("mul3(Tensor input) -> Tensor")


@impl(my_op_lib, "mul3", dispatch_key="CompositeExplicitAutograd")
def mul3_impl(a: torch.Tensor) -> torch.Tensor:
    return a * 3


# registering the out variant.
my_op_lib.define("mul3.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)")


@impl(my_op_lib, "mul3.out", dispatch_key="CompositeExplicitAutograd")
def mul3_out_impl(a: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(a)
    out.mul_(3)
    return out


# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul3.default(a)


def _run(cmd, cwd=None):
    subprocess.run(cmd, stdout=sys.stdout, cwd=cwd, check=True)


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    if args.build_op_package:
        if "HEXAGON_SDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable HEXAGON_SDK_ROOT must be set")
        print(f"HEXAGON_SDK_ROOT={os.getenv('HEXAGON_SDK_ROOT')}")

        if "ANDROID_NDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable ANDROID_NDK_ROOT must be set")
        print(f"ANDROID_NDK_ROOT={os.getenv('ANDROID_NDK_ROOT')}")

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    instance = Model()
    pte_filename = "custom_qnn"
    sample_input = (torch.ones(1, 32, 28, 28),)
    workspace = f"/data/local/tmp/executorch/{pte_filename}"

    soc_info = _soc_info_table[getattr(QcomChipset, args.soc_model)]
    arch = soc_info.htp_info.htp_arch

    # op package setup
    xml_path = f"{args.op_package_dir}/config/example_op_package_htp.xml"
    op_package_config = QnnCustomOpPackageBuilder(
        xml_path=xml_path,
        torch_op_name_map={"ExampleCustomOp": torch.ops.my_ops.mul3.default},
    )
    lib_name = f"libQnn{op_package_config.op_package_name}"

    if args.build_op_package:
        _run(["rm", "-rf", "build"], cwd=args.op_package_dir)
        _run(
            ["make", "htp_x86", "htp_aarch64", f"htp_v{arch}"], cwd=args.op_package_dir
        )
        _run(
            [
                "cp",
                f"{args.op_package_dir}/build/hexagon-v{arch}/{lib_name}.so",
                f"{args.op_package_dir}/build/hexagon-v{arch}/{lib_name}_HTP.so",
            ]
        )

    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.HTP,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=f"{workspace}/{lib_name}_HTP.so",
    )
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=f"{workspace}/{lib_name}.so",
    )
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.X86_64,
        op_package_path=os.path.abspath(
            f"{args.op_package_dir}/build/x86_64-linux-clang/{lib_name}.so"
        ),
    )
    op_package_options = op_package_config.get_op_package_options()
    op_package_paths = [
        f"{args.op_package_dir}/build/hexagon-v{arch}/{lib_name}_HTP.so",
        f"{args.op_package_dir}/build/aarch64-android/{lib_name}.so",
    ]

    # Quantization
    quant_dtype = QuantDtype.use_8a8w
    if args.use_fp16:
        quantizer = None
    else:
        quant_cfg = get_ptq_per_channel_quant_config()
        custom_quant_annotator = CustomOpsQuantAnnotator()
        custom_quant_annotator.register_annotation(
            torch.ops.my_ops.mul3.default,
            IOQuantConfig(
                input_quant_specs={0: quant_cfg.input_activation},
                output_quant_specs={0: quant_cfg.output_activation},
            ),
        )
        annotate_fn = custom_quant_annotator.build_annotation_fn()
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            custom_annotations=(annotate_fn,),
            backend=get_backend_type(args.backend),
            soc_model=args.soc_model,
        )

    build_executorch_binary(
        model=instance,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=[sample_input],
        op_package_options=op_package_options,
        quant_dtype=quant_dtype,
        custom_quantizer=quantizer,
    )

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    if args.enable_x86_64:
        input_list_filename = "input_list.txt"
        generate_inputs(args.artifact, input_list_filename, sample_input)
        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
        target = "x86_64-linux-clang"
        build_folder = os.path.abspath(args.build_folder)
        artifact = os.path.abspath(args.artifact)

        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{build_folder}/lib &&",
                f"{build_folder}/examples/qualcomm/executor_runner/qnn_executor_runner",
                f"--model_path {artifact}/{pte_filename}.pte",
                f"--input_list_path {artifact}/{input_list_filename}",
                f"--output_folder_path {artifact}/outputs",
            ]
        )
        subprocess.run(
            runner_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=artifact,
        )
    else:
        # setup required params accordingly
        # qnn_config    : QnnConfig that saves config info
        # device_id     : serial number of android device
        # workspace     : folder for storing artifacts on android device
        adb = SimpleADB(
            qnn_config=qnn_config,
            pte_path=f"{args.artifact}/{pte_filename}.pte",
            workspace=workspace,
        )
        adb.push(inputs=sample_input, files=op_package_paths)
        if args.debug:
            adb.execute(custom_runner_cmd="logcat -c")
            adb.execute(
                custom_runner_cmd=f"echo 0x1f > {workspace}/qnn_executor_runner.farf"
            )

        adb.execute()
        if args.debug:
            adb.execute(
                custom_runner_cmd=f"logcat -d -v time >{workspace}/outputs/debug_logs.txt"
            )
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
        help="Path to operator package generated from QNN.",
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
        "And add clang compiler into `PATH`. Please refer to Qualcomm AI Engine "
        "Direct SDK document to get more details",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--debug",
        help="Enable device logging",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
