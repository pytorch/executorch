# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import os
import subprocess
import sys
from pathlib import Path

from executorch.backends.nxp.tests.config_importer import test_config

# noinspection PyProtectedMember
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir.schema import DelegateCall, KernelCall


EXECUTORCH_ROOT = test_config.PROJECT_DIR
CMD_TIMEOUT = 300


@contextlib.contextmanager
def _cleanup_generated_files(
    pte_file: Path | None = None, etrecord_file: Path | None = None
):
    """Delete the given generated files once the block finishes, whether the test passed or failed."""
    try:
        yield

    finally:
        if pte_file is not None and pte_file.exists():
            pte_file.unlink()
        if etrecord_file is not None and etrecord_file.exists():
            etrecord_file.unlink()
            parent = etrecord_file.parent
            if not any(parent.iterdir()):
                parent.rmdir()


def _run_compile(cmd: str):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=CMD_TIMEOUT,  # 5 minute timeout just in case. On 8-core x86 the test usually runs ~1 minute.
        cwd=str(EXECUTORCH_ROOT),  # Run from executorch root (like run_aot_example.sh)
    )

    return result


def _assert_delegation(
    process_result: subprocess.CompletedProcess[str], pte_path: Path
):
    # Check script ran successfully
    assert process_result.returncode == 0, (
        f"Script failed with return code {process_result.returncode}\n"
        f"STDOUT:\n{process_result.stdout}\n"
        f"STDERR:\n{process_result.stderr}"
    )

    # Expected .pte file path
    assert pte_path.exists(), f"PTE file not created at {pte_path}"

    # Load and inspect the program to verify delegation
    with open(pte_path, "rb") as f:
        pte_data = f.read()

    program = _deserialize_pte_binary(pte_data).program

    # 1 execution plan (forward).
    assert len(program.execution_plan) == 1
    assert (forward := program.execution_plan[0]).name == "forward"

    # The program only does: Quantize -> Delegate call -> Dequantize
    assert len(ops := forward.operators) == 2  # Quantize and Dequantize
    assert len(forward.chains) == 1
    assert len(instructions := forward.chains[0].instructions) == 3
    # Quantize (Can only check by string. There is no object.)
    assert isinstance(instructions[0].instr_args, KernelCall)
    assert (
        instructions[0].instr_args.op_index == (q_idx := 0)
        and ops[q_idx].name == "quantized_decomposed::quantize_per_tensor"
    )
    # Delegate call
    assert isinstance(instructions[1].instr_args, DelegateCall)
    assert len(forward.delegates) == 1
    assert (
        instructions[1].instr_args.delegate_index == 0
        and forward.delegates[0].id == "NeutronBackend"
    )
    # Dequantize (Can only check by string. There is no object.)
    assert isinstance(instructions[2].instr_args, KernelCall)
    assert (
        instructions[2].instr_args.op_index == (dq_idx := 1)
        and ops[dq_idx].name == "quantized_decomposed::dequantize_per_tensor"
    )


def _assert_profiling(
    process_result: subprocess.CompletedProcess[str],
    pte_path: Path,
    etrecord_path: Path,
):
    # Check script ran successfully.
    assert process_result.returncode == 0, (
        f"Script failed with return code {process_result.returncode}\n"
        f"STDOUT:\n{process_result.stdout}\n"
        f"STDERR:\n{process_result.stderr}"
    )

    # Check if delegated model was created and saved.
    assert pte_path.exists(), f"PTE file not created at {pte_path}"

    # Combine stdout and stderr to capture all subprocess output, including logs.
    process_output = process_result.stdout + process_result.stderr

    # Check if nonempty Neutron to Edge map was created.
    assert "Neutron to Edge map was created:" in process_output

    # Check if ETRecord was created and saved.
    assert "The ETRecord for the model was saved to" in process_output
    assert etrecord_path.exists(), f"ETRecord file not created at {etrecord_path}"


def test_aot_example__mobilenet_v2():
    """Test that mobilenet can be lowered to Neutron backend via `aot_neutron_compile.py` and all ops are delegated."""

    # Run the compilation script as a module (like run_aot_example.sh does)
    cmd = [
        sys.executable,
        "-m",
        "examples.nxp.aot_neutron_compile",
        "--model_name",
        "mobilenetv2",
        "--delegate",
        "--quantize",
        "--target",
        "imxrt700",
        "--use_random_dataset",  # Avoid downloading the dataset.
    ]

    # Output file will be created in executorch_root
    pte_file = Path(os.path.join(EXECUTORCH_ROOT, "mobilenetv2_nxp_delegate.pte"))

    with _cleanup_generated_files(pte_file):
        result = _run_compile(cmd)
        _assert_delegation(result, pte_file)


def test_aot_example__mobilenet_v2__profiling():
    """Test that mobilenet_v2 can be lowered to Neutron backend via `aot_neutron_compile.py`, all ops are delegated,
    the output model is profilable and ETRecord is generated properly."""

    # Run the compilation script as a module (like run_aot_example.sh does)
    cmd = [
        sys.executable,
        "-m",
        "examples.nxp.aot_neutron_compile",
        "--model_name",
        "mobilenetv2",
        "--delegate",
        "--quantize",
        "--target",
        "imxrt700",
        "--remove-quant-io-ops",
        "--use_channels_last_dim_order",
        "--use_profiling",  # Generate profilable model and create ETRecord
        "--use_random_dataset",  # Avoid downloading the dataset.
    ]

    # Output files will be created in executorch_root.
    pte_file = Path(
        os.path.join(EXECUTORCH_ROOT, "mobilenetv2_nxp_delegate_profile.pte")
    )
    etrecord_file = Path(
        os.path.join(EXECUTORCH_ROOT, "etrecord", "mobilenetv2_etrecord.bin")
    )

    with _cleanup_generated_files(pte_file, etrecord_file):
        result = _run_compile(cmd)
        _assert_profiling(result, pte_file, etrecord_file)


def test_aot_example__mlperf_tiny_ic():
    """Test that the MLPerf Tiny image classification model (ResNet-8) can be lowered to Neutron backend via
    `aot_neutron_compile.py` and all ops are delegated."""

    # Run the compilation script as a module (like run_aot_example.sh does).
    # The calibration data of this model is generated randomly, so no dataset download is needed.
    cmd = [
        sys.executable,
        "-m",
        "examples.nxp.aot_neutron_compile",
        "--model_name",
        "mlperf_tiny_image_classification",
        "--delegate",
        "--quantize",
        "--target",
        "imxrt700",
        "--use_random_dataset",
    ]

    # Output file will be created in executorch_root
    pte_file = Path(
        os.path.join(
            EXECUTORCH_ROOT, "mlperf_tiny_image_classification_nxp_delegate.pte"
        )
    )

    with _cleanup_generated_files(pte_file):
        result = _run_compile(cmd)
        _assert_delegation(result, pte_file)


def test_aot_example__mlperf_tiny_ic__profiling():
    """Test that the MLPerf Tiny image classification model (ResNet-8) can be lowered to Neutron backend via
    `aot_neutron_compile.py` and all ops are delegated."""

    # Run the compilation script as a module (like run_aot_example.sh does)
    # Channels-last is buggy, so channels-first is used instead
    cmd = [
        sys.executable,
        "-m",
        "examples.nxp.aot_neutron_compile",
        "--model_name",
        "mlperf_tiny_image_classification",
        "--delegate",
        "--quantize",
        "--target",
        "imxrt700",
        "--remove-quant-io-ops",
        "--use_profiling",  # Generate profilable model and create ETRecord
        "--use_random_dataset",  # Avoid downloading the dataset.
    ]

    # Output files will be created in executorch_root.
    pte_file = Path(
        os.path.join(
            EXECUTORCH_ROOT, "mlperf_tiny_image_classification_nxp_delegate_profile.pte"
        )
    )
    etrecord_file = Path(
        os.path.join(
            EXECUTORCH_ROOT, "etrecord", "mlperf_tiny_image_classification_etrecord.bin"
        )
    )

    with _cleanup_generated_files(pte_file, etrecord_file):
        result = _run_compile(cmd)
        _assert_profiling(result, pte_file, etrecord_file)
