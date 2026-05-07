# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import subprocess
import sys
from pathlib import Path

# noinspection PyProtectedMember
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir.schema import DelegateCall, KernelCall


def test_aot_example__mobilenet_v2():
    """Test that mobilenet can be lowered to Neutron backend via `aot_neutron_compile.py` and all ops are delegated."""

    # Find the executorch root directory (5 levels up from this test file)
    executorch_root = Path(__file__).parent.parent.parent.parent.parent
    assert executorch_root.exists(), f"Executorch root not found at {executorch_root}"

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
    pte_file = executorch_root / "mobilenetv2_nxp_delegate.pte"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout just in case. On my machine, the test usually runs ~1 minute.
            cwd=str(
                executorch_root
            ),  # Run from executorch root (like run_aot_example.sh)
        )

        # Check script ran successfully
        assert result.returncode == 0, (
            f"Script failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Expected .pte file path
        assert pte_file.exists(), f"PTE file not created at {pte_file}"

        # Load and inspect the program to verify delegation
        with open(pte_file, "rb") as f:
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

    finally:
        # Clean up the generated file
        if pte_file.exists():
            pte_file.unlink()


def test_aot_example__mobilenet_v2__profiling():
    """Test that mobilenet_v2 can be lowered to Neutron backend via `aot_neutron_compile.py`, all ops are delegated,
    the output model is profilable and ETRecord is generated properly."""

    # Find the executorch root directory (5 levels up from this test file)
    executorch_root = Path(__file__).parent.parent.parent.parent.parent
    assert executorch_root.exists(), f"Executorch root not found at {executorch_root}"

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
        "--use_profiling", # Generate profilable model and create ETRecord
        "--use_random_dataset",  # Avoid downloading the dataset.
    ]

    # Output files will be created in executorch_root.
    pte_file = executorch_root / "mobilenetv2_nxp_delegate_profile.pte"
    etrecord_file = executorch_root / "etrecord/mobilenetv2_etrecord.bin"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout just in case. On my machine, the test usually runs ~1 minute.
            cwd=str(
                executorch_root
            ),  # Run from executorch root (like run_aot_example.sh)
        )

        # Check script ran successfully.
        assert result.returncode == 0, (
            f"Script failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Check if delegated model was created and saved.
        assert pte_file.exists(), f"PTE file not created at {pte_file}"

        # Combine stdout and stderr to capture all subprocess output, including logs.
        process_output = result.stdout + result.stderr
        print(process_output)

        # Check if nonempty Neutron to Edge map was created.
        assert f"Neutron to Edge map was created:" in process_output

        # Check if ETRecord was created and saved.
        assert f"The ETRecord for the model was saved to" in process_output
        assert etrecord_file.exists(), f"ETRecord file not created at {etrecord_file}"

    finally:
        # Clean up the generated files.
        if pte_file.exists():
            pte_file.unlink()
        if etrecord_file.exists():
            etrecord_file.unlink()
            parent = etrecord_file.parent
            if not any(parent.iterdir()):
                parent.rmdir()
