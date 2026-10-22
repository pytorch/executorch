# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of registering a custom operator for the LPAI (eNPU) backend.

The LPAI op package programming model differs from HTP:
  * the op package implements the QNN OpPackage v1.4 interface
  * the same sources are built twice, selected by ``LPAI_INFERENCE_ONLY``:
      - without the define -> host/compiler side library (x86_64), which also
        contains the inference implementation
      - with the define    -> inference only skel for the DSP
        (hexagon-``LPAI_OP_PACKAGE_ARCH``)
  * kernels access tensors through ``QnnLpaiOpPackage_GlobalInfrastructure_t``

Requires Qualcomm AI Engine Direct SDK >= 2.48, which is the first release that
ships ``QnnLpaiOpPackage.h`` and ``share/QNN/OpPackageGenerator/makefiles/LPAI``.
The on-device path needs >= 2.49, because it additionally requires direct mode.
"""

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
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)
from executorch.backends.qualcomm.utils.utils import get_soc_to_lpai_hw_ver_map
from executorch.examples.qualcomm.utils import make_output_dir
from executorch.exir._serialize._program import deserialize_pte_binary
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


# Hexagon target the DSP side of the op package is built for.
#
# This is neither the HTP architecture of the SoC nor its LPAI hardware version:
# on SM8850 the HTP is V81 and the LPAI hardware version is V6, while the op
# package is compiled for hexagon-v79. The three version numbers are
# independent, so this one cannot be derived from --soc_model. v79 is currently
# the only usable value because the QNN SDK ships a single LPAI op package
# makefile, share/QNN/OpPackageGenerator/makefiles/LPAI/Makefile.hexagon-v79,
# which hardcodes QNN_TARGET, -mv79 and the computev79 QuRT headers.
# Keep in sync with default_op_package_arch in
# backends/qualcomm/scripts/sign_library.sh, which only applies when that script
# is invoked by hand; this driver always passes --op_package_arch explicitly.
LPAI_OP_PACKAGE_ARCH = "v79"


# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul3.default(a)


def _assert_custom_op_delegated(pte_path: str, op_prefix: str):
    """
    Check that the custom op really was lowered into the QNN delegate.

    Comparing the output alone does not prove anything here: an op that is not
    delegated falls back to the CPU implementation registered above, which
    computes the same values, so the numbers would still match while the op
    package was never exercised. The lowered program makes the distinction
    explicit - a delegated op leaves no operator entry behind, while an op that
    fell back appears in the execution plan's operator list.

    Args:
        pte_path: Path of the serialized program to inspect.
        op_prefix: Operator name to look for, without the overload suffix
            ("my_ops::mul3"). Matched as a prefix because the name a fallback
            leaves behind depends on which overload was selected.

    Raises:
        RuntimeError: If the graph has no QNN delegate, or if the custom op is
            still being executed by a CPU kernel.
    """
    with open(pte_path, "rb") as pte_file:
        program = deserialize_pte_binary(pte_file.read()).program

    for plan in program.execution_plan:
        operators = [operator.name for operator in plan.operators]
        fell_back = [name for name in operators if name.startswith(op_prefix)]
        if fell_back:
            raise RuntimeError(
                f"{fell_back} was not delegated, it is executed by a CPU kernel "
                f"in method '{plan.name}'. The op package was not exercised "
                f"even if the output happens to match. Operators: {operators}"
            )
        if not plan.delegates:
            raise RuntimeError(
                f"method '{plan.name}' contains no delegate, the whole graph "
                f"fell back to CPU. Operators: {operators}"
            )


def _run(cmd, cwd=None, env=None):
    subprocess.run(cmd, stdout=sys.stdout, cwd=cwd, env=env, check=True)


def _sign_op_package(op_package_dir: str, lpai_arch: str, op_package_arch: str) -> str:
    """
    Code-sign the DSP objects produced by the op package's hexagon target.

    Everything the aDSP loads must be signed, the op package included, so the
    libraries that get pushed to the device are the signed ones rather than the
    build outputs under ``libs/hexagon-<op_package_arch>``.

    Returns:
        ``$QNN_SDK_ROOT/lib/lpai-<lpai_arch>/signed``, where sign_library.sh
        places the signed libraries.
    """
    executorch_root = os.path.dirname(  # <root>
        os.path.dirname(  # examples
            os.path.dirname(  # qualcomm
                os.path.dirname(os.path.abspath(__file__))  # custom_op
            )
        )
    )
    # sign_library.sh calls elfsigner.py through `python`, and elfsigner.py
    # imports `imp`, which was removed in Python 3.12. Put the interpreter
    # running this script first on PATH so that whatever version is known to
    # work here is used instead of the system default.
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [os.path.dirname(sys.executable), env.get("PATH", "")]
    )
    _run(
        [
            "bash",
            f"{executorch_root}/backends/qualcomm/scripts/sign_library.sh",
            "--lpai_arch",
            lpai_arch,
            "--op_package_dir",
            os.path.abspath(op_package_dir),
            "--op_package_arch",
            op_package_arch,
        ],
        env=env,
    )
    return f"{os.getenv('QNN_SDK_ROOT')}/lib/lpai-{lpai_arch}/signed"


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    if args.backend != "lpai":
        raise RuntimeError(
            f"This example targets the LPAI backend, got --backend {args.backend}"
        )

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    instance = Model()
    pte_filename = "custom_qnn_lpai"
    shape = (1, 32, 28, 28)
    # Calibration and inference inputs are separate tensors on purpose. With a
    # single tensor for both, the input always lands at the top of the
    # calibration range (code 255, stored 127 for 8a8w), which leaves the
    # kernel's low-code and saturation paths unreachable. See --inference_value.
    calibration_input = (torch.full(shape, args.calibration_value),)
    sample_input = (torch.full(shape, args.inference_value),)
    workspace = f"/data/local/tmp/executorch/{pte_filename}"

    xml_path = f"{args.op_package_dir}/config/example_op_package_lpai.xml"
    op_package_config = QnnCustomOpPackageBuilder(
        xml_path=xml_path,
        torch_op_name_map={"ExampleCustomOp": torch.ops.my_ops.mul3.default},
    )
    lib_name = f"libQnn{op_package_config.op_package_name}"
    # Both DSP objects carry the kernel and both have to be deployed: the op
    # package itself and libLpaiOpPackageIsland.so, which is the same kernel
    # linked for always-resident (island) memory. Which one the runtime resolves
    # depends on island mode, so deploy both rather than guessing.
    dsp_lib_names = [f"{lib_name}.so", "libLpaiOpPackageIsland.so"]
    # Unless the libraries get signed below, the ones straight out of the build
    # are used, which only load on a device that does not enforce code signing.
    dsp_lib_dir = f"{args.op_package_dir}/libs/hexagon-{LPAI_OP_PACKAGE_ARCH}"

    if args.build_op_package:
        # The x86_64 library contains both the compiler side and the inference
        # side implementation, so it is all that is needed to compile and to run
        # the model through the LPAI x86_64 simulator.
        build_device_targets = not args.enable_x86_64
        if build_device_targets and "HEXAGON_SDK_ROOT" not in os.environ:
            raise RuntimeError(
                "Environment variable HEXAGON_SDK_ROOT must be set to build "
                f"the DSP (hexagon-{LPAI_OP_PACKAGE_ARCH}) inference skel"
            )

        _run(["make", "clean"], cwd=args.op_package_dir)
        # Each target is built with its own invocation because the toolchain
        # overrides are per target and must not leak into the others.
        #
        # lpai_x86 needs CC/CXX forced to gcc/g++: the QNN LPAI headers include
        # <cstdint>, so the C++ compiler must provide the C++ standard library
        # headers, and a bare clang install may not ship libstdc++ headers.
        _run(["make", "lpai_x86", "CC=gcc", "CXX=g++"], cwd=args.op_package_dir)
        if build_device_targets:
            # The hexagon target uses hexagon-clang from $HEXAGON_SDK_ROOT, so
            # it must not have CC/CXX overridden.
            _run(
                ["make", f"lpai_hexagon_{LPAI_OP_PACKAGE_ARCH}"],
                cwd=args.op_package_dir,
            )
            if not args.skip_sign_op_package:
                # A freshly built op package carries unsigned DSP code, which
                # the aDSP refuses to load, so sign it before deploying. The
                # SoC has already been checked against the supported list by
                # QnnConfig above, so this lookup cannot fail here.
                lpai_arch = f"v{get_soc_to_lpai_hw_ver_map()[args.soc_model]}"
                dsp_lib_dir = _sign_op_package(
                    args.op_package_dir, lpai_arch, LPAI_OP_PACKAGE_ARCH
                )

    op_package_paths = []
    # The x86_64 library is always registered: it is what the LPAI simulator
    # runs, and it is also what compiles the graph on the host (preprocess) for
    # an on-device run.
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.LPAI,
        platform=QnnExecuTorchOpPackagePlatform.X86_64,
        op_package_path=os.path.abspath(
            f"{args.op_package_dir}/libs/x86_64-linux-clang/{lib_name}.so"
        ),
    )
    if not args.enable_x86_64:
        # Register the hexagon build for on-device execution, which runs in
        # direct mode: the delegate itself is compiled for Hexagon and runs
        # inside the DSP process, so it calls registerOpPackage() locally rather
        # than forwarding the registration over FastRPC. There is no 64 bit AP
        # process in that path, so the DSP object is registered directly and is
        # resolved by *base name* through ADSP_LIBRARY_PATH, which is why a bare
        # file name is registered here instead of a path.
        op_package_config.register_implementation(
            target=QnnExecuTorchOpPackageTarget.LPAI,
            platform=QnnExecuTorchOpPackagePlatform.HEXAGON,
            op_package_path=f"{lib_name}.so",
        )
        op_package_paths = [f"{dsp_lib_dir}/{name}" for name in dsp_lib_names]
    op_package_options = op_package_config.get_op_package_options()

    # Quantization. The LPAI backend is a fixed point accelerator and the op
    # package declares support for QNN_DATATYPE_UFIXED_POINT_8, so the custom op
    # is annotated with 8 bit activations. 16 bit activations are not supported
    # end to end yet, see the data type comment in
    # example_op_package_lpai/ExampleLpaiOpPackage/src/ops/ExampleCustomOp_compiler.cpp.
    quant_dtype = QuantDtype.use_8a8w
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
        dataset=[calibration_input],
        op_package_options=op_package_options,
        quant_dtype=quant_dtype,
        custom_quantizer=quantizer,
    )

    # The output comparison further down cannot tell a delegated op from one
    # that fell back to the CPU implementation registered above, so check the
    # lowered graph explicitly before running it.
    _assert_custom_op_delegated(f"{args.artifact}/{pte_filename}.pte", "my_ops::mul3")

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
        # check=True so that a runner failure is reported as such: without it
        # the only symptom is the missing output file read further down, which
        # says nothing about what actually went wrong.
        subprocess.run(
            runner_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=artifact,
            check=True,
        )
    else:
        adb = SimpleADB(
            qnn_config=qnn_config,
            pte_path=f"{args.artifact}/{pte_filename}.pte",
            workspace=workspace,
        )
        adb.push(inputs=sample_input, files=op_package_paths)
        if args.debug:
            adb.execute(custom_runner_cmd="logcat -c")
            # FARF configuration is looked up as <runner base name>.farf, and in
            # direct mode the runner is qnn_executor_direct_runner rather than
            # qnn_executor_runner, so derive the name instead of hardcoding it.
            runner_name = os.path.basename(adb.runner)
            adb.execute(custom_runner_cmd=f"echo 0x1f > {workspace}/{runner_name}.farf")

        adb.execute()
        if args.debug:
            adb.execute(
                custom_runner_cmd=f"logcat -d -v time >{workspace}/outputs/debug_logs.txt"
            )
        adb.pull(host_output_path=args.artifact)

    # By default the eager result is the reference. When the inference input
    # exceeds the calibrated range the quantized graph must saturate instead,
    # so --expected_value states that clamped result explicitly.
    x86_golden = (
        instance(*sample_input)
        if args.expected_value is None
        else torch.full(shape, args.expected_value)
    )
    device_output = torch.from_numpy(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
    ).reshape(x86_golden.size())
    # The op package requantizes into 8 bit, so compare with a tolerance that
    # accounts for a few quantization steps (3/255 wide each by default).
    result = torch.all(
        torch.isclose(x86_golden, device_output, atol=args.atol)
    ).tolist()

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "is_close": result,
                        # Reaching this point means the op package was really
                        # used, see _assert_custom_op_delegated().
                        "is_delegated": True,
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
        help="path for storing generated artifacts by this example. "
        "Default ./custom_op_lpai",
        default="./custom_op_lpai",
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
        "--build_op_package",
        help="Build op package based on op_package_dir. Building the DSP skel "
        "additionally requires `HEXAGON_SDK_ROOT` to be set. Please refer to "
        "Qualcomm AI Engine Direct SDK document to get more details",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--skip_sign_op_package",
        help="Do not code-sign the DSP libraries built by --build_op_package. "
        "The unsigned libraries only load on a device which does not enforce "
        "code signing.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--atol",
        help="Absolute tolerance used when comparing against the eager result. "
        "The default leaves room for a few output quantization steps, which are "
        "3/255 wide for the default calibration range in 8 bit.",
        default=0.05,
        type=float,
    )

    parser.add_argument(
        "--calibration_value",
        help="Value of the constant tensor the quantizer is calibrated with. "
        "It fixes the quantization range, so with the default 1.0 the input "
        "scale is 1/255 and the output scale 3/255.",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--inference_value",
        help="Value of the constant tensor the model is run with. Keep it below "
        "--calibration_value to exercise the kernel's low code path (e.g. 0.25 "
        "gives code 64, stored as the byte 192 once biased), or above it to "
        "exercise saturation (e.g. 2.0 requantizes to code 510, which must be "
        "clamped to 255).",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--expected_value",
        help="Compare the output against this constant instead of the eager "
        "result. Needed when --inference_value is outside the calibrated range, "
        "where the correct answer is the saturated one rather than 3x the input.",
        default=None,
        type=float,
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
            raise
