# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess  # nosec B404 - launches trusted build/test scripts
import sys
import time
from typing import Sequence


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
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        required=False,
        default=False,
        help="Don't quantize model",
    )
    parser.add_argument(
        "--extra_flags",
        required=False,
        default="",
        help="Extra cmake flags to pass the when building the executor_runner",
    )
    parser.add_argument(
        "--extra_runtime_flags",
        required=False,
        default="",
        help="Extra runtime flags to pass the final runner/executable",
    )
    parser.add_argument(
        "--timeout",
        required=False,
        default=60 * 20,
        help="Timeout in seconds used when running the model",
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
            args.memory_mode = "Dedicated_Sram_384KB"
        else:
            raise RuntimeError(f"Invalid target name {args.target}")

    return args


def run_external_cmd(cmd: Sequence[str]) -> None:
    print("CALL:", *cmd, sep=" ")
    try:
        subprocess.check_call(
            cmd
        )  # nosec B603 - cmd assembled from vetted scripts/flags
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
            "--devtools",
            "--etdump",
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
    no_quantize: bool,
):
    command_list = [
        "python3",
        "-m",
        "examples.arm.aot_arm_compiler",
        "--delegate",
        "--bundleio",
        f"--model_name={model_name}",
        f"--target={target}",
        f"--output={build_output}",
    ]

    if "vgf" != target:
        command_list.append(f"--system_config={system_config}")
        command_list.append(f"--memory_mode={memory_mode}")

    if not no_quantize:
        command_list.append("--quantize")

    if not no_intermediate:
        command_list.append(f"--intermediate={output}")

    run_external_cmd(command_list)

    pte_file_ending = "bpte"
    pte_file = os.path.join(
        output, f"{model_name}_arm_delegate_{args.target}.{pte_file_ending}"
    )

    return pte_file


def build_ethosu_runtime(
    et_build_root: str,
    script_path: str,
    pte_file: str,
    target: str,
    system_config: str,
    memory_mode: str,
    extra_flags: str,
    elf_build_path: str,
):
    elf_build_path = os.path.join(elf_build_path, "cmake-out")
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "build_executor_runner.sh"),
            f"--et_build_root={et_build_root}",
            f"--pte={pte_file}",
            "--bundleio",
            "--etdump",
            f"--target={target}",
            "--build_type=Release",
            f"--system_config={system_config}",
            f"--memory_mode={memory_mode}",
            f"--extra_build_flags=-DET_LOG_DUMP_OUTPUT=OFF {extra_flags}",
            f"--output={elf_build_path}",
        ]
    )

    elf_file = os.path.join(elf_build_path, "arm_executor_runner")
    return elf_file


def run_elf_with_fvp(script_path: str, elf_file: str, target: str, timeout: int):
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "run_fvp.sh"),
            f"--elf={elf_file}",
            f"--target={target}",
            f"--timeout={timeout}",
        ]
    )


def build_vkml_runtime(
    et_build_root: str,
    script_path: str,
    extra_flags: str,
    build_path: str,
):
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "build_executor_runner_vkml.sh"),
            f"--et_build_root={et_build_root}",
            "--etdump",
            "--bundleio",
            "--build_type=Release",
            f"--extra_build_flags=-DET_DUMP_OUTPUT=OFF {extra_flags}",
            f"--output={build_path}",
        ]
    )

    runner = os.path.join(build_path, "executor_runner")
    return runner


def run_vkml(script_path: str, pte_file: str, runner_build_path: str, extra_flags: str):
    run_external_cmd(
        [
            "bash",
            os.path.join(script_path, "run_vkml.sh"),
            f"--model={pte_file}",
            f"--build_path={runner_build_path}",
            f"--optional_flags={extra_flags}",
        ]
    )


if __name__ == "__main__":
    total_start_time = time.perf_counter()
    args = get_args()
    script_path = os.path.join("backends", "arm", "scripts")

    if args.build_libs:
        start_time = time.perf_counter()
        build_libs(args.test_output, script_path)
        end_time = time.perf_counter()
        print(
            f"[Test model: {end_time - start_time:.2f} s] Build needed executorch libs"
        )

    if args.model:
        model_name = args.model.split(" ")[0].split(";")[0]
        if not model_name:
            print("ERROR: Bad --model specified")
        if not args.target:
            print("ERROR: --model need --target to also be set")

        output = os.path.join(
            args.test_output, f"{model_name}_arm_delegate_{args.target}"
        )

        start_time = time.perf_counter()
        pte_file = build_pte(
            args.test_output,
            model_name,
            args.target,
            args.system_config,
            args.memory_mode,
            output,
            args.no_intermediate,
            args.no_quantize,
        )
        end_time = time.perf_counter()
        print(
            f"[Test model: {end_time - start_time:.2f} s] PTE file created: {pte_file}"
        )

        if "vgf" == args.target:
            build_path = os.path.join(
                output, f"{model_name}_arm_delegate_{args.target}"
            )

            start_time = time.perf_counter()
            vkml_runner = build_vkml_runtime(
                args.test_output,
                script_path,
                args.extra_flags,
                build_path,
            )
            end_time = time.perf_counter()
            print(
                f"[Test model: {end_time - start_time:.2f} s] ELF file created: {vkml_runner}"
            )

            start_time = time.perf_counter()
            run_vkml(script_path, pte_file, build_path, args.extra_runtime_flags)
            end_time = time.perf_counter()
            print(
                f"[Test model: {end_time - start_time:.2f} s] Tested VKML runner: {vkml_runner}"
            )

        elif "ethos-u" in args.target:
            elf_build_path = os.path.join(
                output, f"{model_name}_arm_delegate_{args.target}"
            )

            start_time = time.perf_counter()
            elf_file = build_ethosu_runtime(
                args.test_output,
                script_path,
                pte_file,
                args.target,
                args.system_config,
                args.memory_mode,
                args.extra_flags,
                elf_build_path,
            )
            end_time = time.perf_counter()
            print(
                f"[Test model: {end_time - start_time:.2f} s] ELF file created: {elf_file}"
            )

            start_time = time.perf_counter()
            run_elf_with_fvp(script_path, elf_file, args.target, args.timeout)
            end_time = time.perf_counter()
            print(
                f"[Test model: {end_time - start_time:.2f} s] Tested elf on FVP {elf_file}"
            )
        total_end_time = time.perf_counter()
        print(
            f"[Test model: {total_end_time - total_start_time:.2f} s total] Model: {model_name} on {args.target} -> PASS"
        )
