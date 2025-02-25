# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import platform
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build_libs",
        action="store_true",
        required=False,
        default=False,
        help="Flag for building executorch libs needed for this testing",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Model to use that aot_arm_compiler.py can handle, can be a builtin, examples/models or a filename.",
    )
    parser.add_argument(
        "--target",
        required=False,
        default=None,
        help="Target name",
    )
    parser.add_argument(
        "--test_output",
        required=False,
        default="arm_test",
        help="Output folder used for build and test defults to arm_test",
    )
    parser.add_argument(
        "--system_config",
        required=False,
        default=None,
        help="Target specific system_config (See Vela compiler)",
    )
    parser.add_argument(
        "--memory_mode",
        required=False,
        default=None,
        help="Target specific memory_mode (See Vela compiler)",
    )
    parser.add_argument(
        "--no_intermediate",
        action="store_true",
        required=False,
        default=False,
        help="Don't save temporary files during compilation",
    )

    args = parser.parse_args()

    if args.model and "ethos-u" in args.target and args.system_config is None:
        if "u55" in args.target:
            args.system_config = "Ethos_U55_High_End_Embedded"
        elif "u85" in args.target:
            args.system_config = "Ethos_U85_SYS_DRAM_Mid"
        else:
            raise RuntimeError(f"Invalid target name {args.target}")

    if args.model and "ethos-u" in args.target and args.memory_mode is None:
        if "u55" in args.target:
            args.memory_mode = "Shared_Sram"
        elif "u85" in args.target:
            args.memory_mode = "Sram_Only"
        else:
            raise RuntimeError(f"Invalid target name {args.target}")

    return args


def run_external_cmd(cmd: []):
    print("CALL:", *cmd, sep=" ")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as err:
        print("ERROR called: ", *cmd, sep=" ")
        print(f"Failed with: {err.returncode}")
        sys.exit(err.returncode)


def build_libs(et_build_root: str, script_path: str):
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "build_executorch.sh"),
            f"--et_build_root={et_build_root}",
            "--build_type=Release",
        ]
    )
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "build_portable_kernels.sh"),
            f"--et_build_root={et_build_root}",
            "--build_type=Release",
            "--portable_kernels=aten::_softmax.out",
        ]
    )
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "build_quantized_ops_aot_lib.sh"),
            f"--et_build_root={et_build_root}",
            "--build_type=Release",
        ]
    )


def build_pte(
    et_build_root: str,
    model_name: str,
    target: str,
    system_config: str,
    memory_mode: str,
    build_output: str,
    no_intermediate: bool,
):
    soext = {"Darwin": "dylib", "Linux": "so", "Windows": "dll"}.get(
        platform.system(), None
    )
    solibs_path = os.path.join(
        et_build_root,
        "cmake-out-aot-lib",
        "kernels",
        "quantized",
        f"libquantized_ops_aot_lib.{soext}",
    )
    solibs = f"--so_library={solibs_path}"

    intermediate = ""
    if not no_intermediate:
        intermediate = f"--intermediate={output}"

    run_external_cmd(
        [
            "python3",
            "-m",
            "examples.arm.aot_arm_compiler",
            "--delegate",
            "--quantize",
            intermediate,
            f"--model_name={model_name}",
            f"--target={target}",
            f"--output={build_output}",
            f"--system_config={system_config}",
            f"--memory_mode={memory_mode}",
            solibs,
        ]
    )

    pte_file = os.path.join(output, f"{model_name}_arm_delegate_{args.target}.pte")
    return pte_file


def build_ethosu_runtime(
    et_build_root: str,
    script_path: str,
    pte_file: str,
    target: str,
    system_config: str,
    elf_build_path: str,
):
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "build_executorch_runner.sh"),
            f"--et_build_root={et_build_root}",
            f"--pte={pte_file}",
            f"--target={target}",
            "--build_type=Release",
            f"--system_config={system_config}",
            f"--output={elf_build_path}",
        ]
    )

    elf_file = os.path.join(elf_build_path, "cmake-out", "arm_executor_runner")
    return elf_file


def run_elf_with_fvp(script_path: str, elf_file: str, target: str):
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "run_fvp.sh"),
            f"--elf={elf_file}",
            f"--target={target}",
        ]
    )


if __name__ == "__main__":

    args = get_args()
    script_path = os.path.join("backends", "arm", "scripts")

    if args.build_libs:
        build_libs(args.test_output, script_path)

    if args.model:
        model_name = args.model.split(" ")[0].split(";")[0]
        if not model_name:
            print("ERROR: Bad --model specified")
        if not args.target:
            print("ERROR: --model need --target to also be set")

        output = os.path.join(
            args.test_output, f"{model_name}_arm_delegate_{args.target}"
        )

        pte_file = build_pte(
            args.test_output,
            model_name,
            args.target,
            args.system_config,
            args.memory_mode,
            output,
            args.no_intermediate,
        )
        print(f"PTE file created: {pte_file} ")

        if "ethos-u" in args.target:
            elf_build_path = os.path.join(
                output, f"{model_name}_arm_delegate_{args.target}"
            )

            elf_file = build_ethosu_runtime(
                args.test_output,
                script_path,
                pte_file,
                args.target,
                args.system_config,
                elf_build_path,
            )
            print(f"ELF file created: {elf_file} ")

            run_elf_with_fvp(script_path, elf_file, args.target)
        print(f"Model: {model_name} on {args.target} -> PASS")
