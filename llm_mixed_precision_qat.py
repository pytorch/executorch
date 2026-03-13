import json
import os

import torch
from torchao.quantization.pt2e import MovingAverageMinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.utils.utils import draw_graph
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import LLM_VARIANT_ARCHS
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.utils import (
    make_quantizer,
    setup_common_args_and_variables,
    QuantDtype,
)
from pytorch_tokenizers import get_tokenizer

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
    # 2. Model Creation
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

    # ====================================================================
    # 3. Quantization Annotation
    # ====================================================================
    use_qat, use_verbose = True, True
    quantizer = make_quantizer(is_qat=use_qat)
    quantizer.recipe = QuantRecipe(
        # basic dtype
        quant_dtype=QuantDtype.use_16a4w,
        is_qat=use_qat,
        act_observer=MovingAverageMinMaxObserver,
        granularity=QuantGranularity.PER_TENSOR,
        verbose=use_verbose,
    ).add_node_target(
        # annotate all attention layers with 16a4w lpbq
        targets={
            torch.ops.aten.linear.default,
        },
        quant_dtype=QuantDtype.use_16a4w_block,
        is_qat=use_qat,
        act_observer=MovingAverageMinMaxObserver,
        granularity=QuantGranularity.PER_BLOCK,
        extra_kwargs={"block_size": (1, 64, 1, 1)},
    ).add_regex(
        # annotate down_proj layer with 16a8w pcq
        regex={r"layers\..*\.feed_forward\.w2",},
        quant_dtype=QuantDtype.use_16a8w,
        is_qat=use_qat,
        act_observer=MovingAverageMinMaxObserver,
        granularity=QuantGranularity.PER_CHANNEL,
    )

    model = get_quant_embedding_transform("4,32")(model)
    module = torch.export.export(model, input).module()
    module = prepare_pt2e(module, quantizer)
    draw_graph("llm_qat", ".", module)


if __name__ == "__main__":
    main()  # pragma: no cover
