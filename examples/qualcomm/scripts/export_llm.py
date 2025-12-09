# pyre-ignore-all-errors
"""
Export LLM Script for Qualcomm Backend

This script exports a LLaMA-style language model to ExecuTorch format optimized
for Qualcomm's QNN (Qualcomm Neural Network) backend. It supports both Post-Training
Quantization (PTQ) and Quantization-Aware Training (QAT) workflows.


Usage:
    python export_llm.py --checkpoint <path> --params <path> --tokenizer_model <path> \
        --prompt "Once upon a time" --quantization ptq --output_folder /tmp -s <serial>
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import getpass
import json
import os

import numpy as np
import torch
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.utils import SimpleADB
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.export_util.utils import save_pte_program
from pytorch_tokenizers import get_tokenizer
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)


def compute_snr(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) between two tensors.
    """
    assert x.shape == y.shape, f"Tensor shapes do not match {x.shape} vs {y.shape}"
    x = x.float()
    y = y.float()
    error = x - y
    original_power = torch.mean(torch.pow(x, 2))
    error_power = torch.mean(torch.pow(error, 2))
    snr = 10 * torch.log10(original_power / error_power)
    return round(snr.item(), 2)


def get_stories110m_model(args):
    """
    Create and configure a stories110m model.

    Args:
        args: Command-line arguments containing:
            - params: Path to the model parameters JSON file
            - max_seq_len: Maximum sequence length to process
    """
    # Load model configuration from JSON params file
    params_path = args.params
    with open(params_path) as f:
        prefill_config = ModelArgs(**json.load(f))

    prefill_config.max_batch_size = 1
    prefill_config.max_seq_len = args.max_seq_len
    prefill_config.use_kv_cache = False

    return LLM_VARIANT_ARCHS.get("stories110m", LlamaModel)(
        prefill_config,
        ar_len=args.max_seq_len,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
    )


def main() -> None:
    """
    Main function that orchestrates the LLM export pipeline.

    This function performs the following steps:
    1. Parse command-line arguments
    2. Tokenize input prompt
    3. Initialize and configure the model
    4. Quantize the model (PTQ or QAT)
    5. Compile for Qualcomm QNN backend
    6. Export to .pte format
    7. Run inference on target platform (Android)
    8. Compare outputs and compute SNR
    """
    parser = argparse.ArgumentParser()

    # Required: Path to model checkpoint weights
    parser.add_argument(
        "--checkpoint",
        help="Pass llama checkpoint.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama params json file.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="",
        help="The folder to store the exported program",
    )

    parser.add_argument(
        "--soc",
        type=str,
        default="SM8650",
        help="Specify the SoC model.",
    )

    parser.add_argument(
        "--quantization",
        choices=["ptq", "qat"],
        help="Run post-traininig quantization.",
    )
    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prompt",
        help="User prompts for Llama.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--max_seq_len",
        help="This refers to maximum number of tokens that the model can process & consider at once to generate predictions/responses.",
        default=128,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="serial number for android device communicated via ADB.",
        type=str,
    )
    args = parser.parse_args()

    # ====================================================================
    # 1. Example Inputs Preparation
    # ====================================================================
    tokenizer = get_tokenizer(args.tokenizer_model)
    token_list = tokenizer.encode(args.prompt, bos=True, eos=False)

    # Convert tokens to tensor and truncate to max_seq_len if necessary
    token_tensor = torch.tensor(token_list, dtype=torch.int32)[: args.max_seq_len]

    # Pad token tensor to max_seq_len with zeros
    token_tensor = torch.cat(
        [
            token_tensor.unsqueeze(0),  # Resize for batch dimension
            torch.zeros((1, args.max_seq_len - len(token_list)), dtype=torch.int32),
        ],
        dim=1,
    )

    # ====================================================================
    # 2. Model Creation and Quantization
    # ====================================================================
    model = get_stories110m_model(args)
    state_dict = torch.load(
        args.checkpoint, weights_only=True, map_location="cpu", mmap=True
    )
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(
        state_dict,
        strict=True,
        assign=True,
    )
    _, atten_mask = model.get_example_inputs(use_kv_cache=False)
    example_inputs = (
        token_tensor,
        atten_mask.masks[0].mask.to(torch.float32),
    )
    print(
        f"example inputs: tokens {example_inputs[0].shape}, mask {example_inputs[1].shape}"
    )

    print("Quantizing model...")
    # It is the model quantization path
    quantizer = QnnQuantizer()
    # Typical pytorch 2.0 quantization flow
    m = torch.export.export(model.eval(), example_inputs, strict=True).module()
    if args.quantization == "qat":
        m = prepare_qat_pt2e(m, quantizer)
        # Training loop
        m(*example_inputs)
    elif args.quantization == "ptq":
        m = prepare_pt2e(m, quantizer)
        # Calibration
        m(*example_inputs)
    else:
        raise RuntimeError(f"Unknown quantization type {args.quantization}")
    m = convert_pt2e(m)

    # Get the quantized model outputs for future comparison
    quantized_output = m(*example_inputs)
    quantized_output_logits = quantized_output[0]  # Extract logits from output tuple
    print(f"quantized output: {quantized_output_logits}")

    # ====================================================================
    # 3. Model Lowering
    # ====================================================================
    backend_options = generate_htp_compiler_spec(
        use_fp16=False,
    )
    # TODO: disable memory plan
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[args.soc],
        backend_options=backend_options,
    )
    delegated_program = to_edge_transform_and_lower_to_qnn(
        m,
        example_inputs,
        compile_spec,
    )
    executorch_config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        ),
        extract_delegate_segments=True,
    )
    executorch_program = delegated_program.to_executorch(executorch_config)
    pte_path = save_pte_program(executorch_program, "llm", args.output_folder)

    # Save input tensors as binary files for runtime execution
    input_files = []
    for i, input in enumerate(example_inputs):
        input_path = os.path.join(args.output_folder, f"input_{i}.pt")
        input.numpy().tofile(input_path)
        input_files.append(f"input_{i}.pt")

    # Create input list file that tells the runner which inputs to use
    input_list_path = os.path.join(args.output_folder, "input_list.txt")

    # ====================================================================
    # 4. On Device Execution
    # ====================================================================
    # Write input file names (relative paths for Android execution)
    with open(input_list_path, "w", encoding="utf-8") as f:
        f.write(" ".join(input_files))

    print(f"inputs are saved to {input_list_path}")
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            f"./qnn_executor_runner",
            f"--model_path llm.pte",
            f"--input_list_path input_list.txt",
            f"--output_folder_path outputs",
        ]
    )
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path="build-android",
        pte_path=pte_path,
        workspace=workspace,
        device_id=args.device,
        soc_model=args.soc,
        runner=f"examples/qualcomm/executor_runner/qnn_executor_runner",
    )
    adb.push(
        inputs=[],
        files=[input_list_path, pte_path]
        + [os.path.join(args.output_folder, input_file) for input_file in input_files],
    )
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Running command: {runner_cmd}")
    adb.execute(custom_runner_cmd=runner_cmd)
    adb.pull(output_path=args.output_folder)

    # ====================================================================
    # 5. SNR Computation
    # ====================================================================
    logits = torch.tensor(
        np.fromfile(f"{args.output_folder}/outputs/output_0_0.raw", dtype=np.float32)
    )
    print(f"pte output: {logits}")
    print(f"snr: {compute_snr(torch.flatten(quantized_output_logits), logits)}")


if __name__ == "__main__":
    main()  # pragma: no cover
