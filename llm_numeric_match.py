import json
import os

import torch
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.devtools import Inspector
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_quantizer,
    setup_common_args_and_variables,
    QuantDtype,
    SimpleADB,
)
from pytorch_tokenizers import get_tokenizer
from executorch.backends.qualcomm.debugger.metrics_evaluator import (
    MetricEvaluatorBase,
    CosineSimilarityEvaluator,
)
from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
    OutputFormat,
    QNNIntermediateDebugger,
)
from torchao.quantization.utils import compute_error


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
    # please change here for further analysis
    prefill_config.n_layers = 1

    model = LLM_VARIANT_ARCHS.get("stories110m", LlamaModel)(
        prefill_config,
        ar_len=args.max_seq_len,
        output_new_cache_only=False,
        output_cache=False,
        use_i64_token=False,
    )
    return model


def main() -> None:
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./llm_debug",
        default="./llm_debug",
        type=str,
    )
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
    args = parser.parse_args()

    # ====================================================================
    # 1. Example Inputs Preparation
    # ====================================================================
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)
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
    # 2. Model Creation and Lowering
    # ====================================================================
    model = get_stories110m_model(args)
    state_dict = torch.load(
        args.checkpoint, weights_only=True, map_location="cpu", mmap=True
    )
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(
        state_dict,
        strict=False,
        assign=True,
    )
    _, atten_mask = model.get_example_inputs()
    input = (
        token_tensor,
        atten_mask.masks[0].mask.to(torch.float32),
    )

    qnn_intermediate_debugger = QNNIntermediateDebugger()
    pte_filename = "llm"
    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a4w,
        per_channel_linear=True,
    )
    build_executorch_binary(
        model,
        input,
        args.model,
        f"{args.artifact}/{pte_filename}",
        [input],
        custom_quantizer=quantizer,
        dump_intermediate_outputs=True,
        qnn_intermediate_debugger=qnn_intermediate_debugger,
    )

    # ====================================================================
    # 3. On Device Execution & Accuracy Analysis
    # ====================================================================
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        dump_intermediate_outputs=True,
        # reserve some margin for nodes added in preprocess
        debug_buffer_size=int(qnn_intermediate_debugger.debug_buffer_size * 3),
    )
    adb.push(inputs=[input])
    adb.execute()

    class SqnrEvaluator(MetricEvaluatorBase):
      def __init__(self, threshold=20.0):
          self.threshold = threshold

      def metric_name(self) -> str:
          return "SQNR"

      def evaluate(
          self, qnn_output: torch.Tensor, cpu_output: torch.Tensor
      ):
          sqnr = compute_error(cpu_output, qnn_output)
          valid = sqnr >= self.threshold
          return sqnr, valid

    def validate_intermediate_tensor():
        inspector = Inspector(
            etdump_path=f"{args.artifact}/etdump.etdp",
            debug_buffer_path=f"{args.artifact}/debug_output.bin",
        )

        edge_output = qnn_intermediate_debugger.intermediate_output_module(
            input[1], input[0]
        )
        numel = [t.numel() for t in edge_output]
        edge_result = edge_output[numel.index(max(numel))]

        # Optional: Ensures that edge module accuracy aligns with nn.Module
        with torch.no_grad():
            source_result = model(*input)[0]
            score = torch.nn.functional.cosine_similarity(
                edge_result.flatten(), source_result.flatten(), dim=0
            ).item()
            print("Cosine Similarity Score between nn.Module and Edge CPU is: ", score)
            score = compute_error(edge_result.flatten(), source_result.flatten())
            print("SQNR Score between nn.Module and Edge CPU is: ", score)

        # Users can generate multiple comparison metrics in a single execution.
        # Below, we generate 3 metrics.
        qnn_intermediate_debugger.generate_results(
            title="llm_cos_similarity_debugging_graph",
            path=args.artifact,
            output_format=OutputFormat.SVG_GRAPHS,
            inspector=inspector,
            evaluator=CosineSimilarityEvaluator(0.9),
        )
        qnn_intermediate_debugger.generate_results(
            title="llm_cos_similarity_csv",
            path=args.artifact,
            output_format=OutputFormat.CSV_FILES,
            inspector=inspector,
            evaluator=CosineSimilarityEvaluator(0.9),
        )
        # Using self defined metrics to print svg graphs
        qnn_intermediate_debugger.generate_results(
            title="llm_sqnr_debugging_graph",
            path=args.artifact,
            output_format=OutputFormat.SVG_GRAPHS,
            inspector=inspector,
            evaluator=SqnrEvaluator(20.0),
        )

    adb.pull_debug_output(
        args.artifact, args.artifact, callback=validate_intermediate_tensor
    )


if __name__ == "__main__":
    main()  # pragma: no cover
