# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side static llama QDQ evaluation for QNN quantization.

This entrypoint is used by the QNN static llama eval CI job. It evaluates the
eager exported graph and the converted QDQ graph locally with lm-eval, without
requiring a connected Qualcomm device.
"""

# pyre-ignore-all-errors

import json
import logging
import os
import sys
import types
from typing import Callable

import torch

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
from executorch.examples.models.llama.eval_llama_lib import build_args_parser
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.range_setting_pt2e import (
    compute_scales,
    make_custom_quantizer,
    reverse_quantize_module_swap,
    set_scales,
    WrappedLlamaModel,
)
from lm_eval.evaluator import simple_evaluate
from pytorch_tokenizers import get_tokenizer, TiktokenTokenizer
from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from torchao.prototype.spinquant import apply_spinquant
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoTokenizer


sys.setrecursionlimit(4096)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


class StaticPrefillEvalWrapper(EagerEvalWrapper):
    """lm-eval wrapper for QNN static prefill graphs with fixed input shapes."""

    def __init__(
        self,
        model: torch.fx.GraphModule,
        tokenizer: HuggingFaceTokenizer | SentencePieceTokenizer | TiktokenTokenizer,
        get_example_inputs: Callable,
        max_seq_length: int,
        use_i64_token: bool,
        device: str,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            device=device,
        )
        self._model = model.to(self.device)
        _, self._atten_mask = get_example_inputs()
        self._token_dtype = torch.int64 if use_i64_token else torch.int32

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        seq_len = min(inps.shape[-1], self._max_seq_length)
        tokens = inps[:, :seq_len].to(device=self.device, dtype=self._token_dtype)
        if seq_len < self._max_seq_length:
            tokens = torch.nn.functional.pad(
                tokens, (0, self._max_seq_length - seq_len)
            )

        results = self._model(tokens, *self._atten_mask)
        logits = results[0] if isinstance(results, (tuple, list)) else results
        return logits[:, :seq_len, :]

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError("Generation is not used by lm-eval perplexity tasks")


def prepare_tokenizer(args):
    if args.decoder_model in {"stories110m", "stories260k"}:
        tokenizer = get_tokenizer(args.tokenizer_model)
        assert isinstance(
            tokenizer, SentencePieceTokenizer
        ), "Wrong tokenizer provided for stories."
        assert (
            args.tokenizer_bin is not None
        ), "Please provide tokenizer_bin for stories."
    elif "llama3_2" in args.decoder_model:
        tokenizer = get_tokenizer(args.tokenizer_model)
        assert isinstance(
            tokenizer, TiktokenTokenizer
        ), "Wrong tokenizer provided for llama3_2."
    elif args.decoder_model in SUPPORTED_LLM_MODELS:
        model_id = SUPPORTED_LLM_MODELS[args.decoder_model].repo_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        runtime_tokenizer_path = tokenizer.save_pretrained(args.artifact)[-1]
        tokenizer = get_tokenizer(runtime_tokenizer_path)
    else:
        raise RuntimeError(f"Unknown decoder_model: {args.decoder_model}.")
    return tokenizer


def prepare_model(args):
    params_path = args.params or SUPPORTED_LLM_MODELS[args.decoder_model].params_path
    with open(params_path) as f:
        prefill_config = ModelArgs(**json.load(f))

    prefill_config.max_batch_size = 1
    prefill_config.max_seq_len = args.max_seq_length
    prefill_config.max_context_len = args.max_seq_length
    prefill_config.use_kv_cache = False
    prefill_config.enable_r3 = args.r3

    use_i64_token = args.embedding_quantize is not None
    model = LlamaModel(
        prefill_config,
        ar_len=args.max_seq_length,
        output_new_cache_only=True,
        output_cache=False,
        use_i64_token=use_i64_token,
    )

    if args.checkpoint is None:
        model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
        checkpoint = download_and_convert_hf_checkpoint(
            model_config.repo_id,
            model_config.convert_weights.__func__,
        )
        state_dict = torch.load(
            checkpoint, weights_only=True, map_location=args.device, mmap=True
        )
        transform_weight = model_config.transform_weight
    else:
        state_dict = torch.load(
            args.checkpoint, weights_only=True, map_location=args.device, mmap=True
        )
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if args.decoder_model == "stories260k":
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        transform_weight = True

    if transform_weight:
        n_heads = model.n_heads
        n_kv_heads = model.n_kv_heads
        for layer_i in range(model.n_layers):
            state_dict[f"layers.{layer_i}.attention.wq.weight"] = _permute_rope_weight(
                state_dict[f"layers.{layer_i}.attention.wq.weight"], n_heads
            )
            state_dict[f"layers.{layer_i}.attention.wk.weight"] = _permute_rope_weight(
                state_dict[f"layers.{layer_i}.attention.wk.weight"], n_kv_heads
            )

    model.load_state_dict(state_dict, strict=True, assign=True)
    return model, prefill_config


def _permute_rope_weight(weight: torch.Tensor, heads: int) -> torch.Tensor:
    dim_0 = weight.size(0)
    dim_1 = weight.size(1)
    return (
        weight.view(heads, dim_0 // heads // 2, 2, dim_1)
        .transpose(1, 2)
        .reshape(dim_0, dim_1)
    )


def prequant_algorithm(model, prefill_config, args):
    model = model.to(device=args.device, dtype=torch.float)
    inputs = model.get_example_inputs()
    tokens, atten_mask = inputs
    tokens.to(args.device)
    for mask in atten_mask.masks:
        mask.mask.to(args.device)

    scales_state_dict = {}

    if args.spinquant:
        config = types.SimpleNamespace(
            dim=prefill_config.dim,
            head_dim=prefill_config.dim // prefill_config.n_heads,
            n_local_heads=prefill_config.n_heads,
            intermediate_size=4 * prefill_config.dim,
        )
        model.config = config
        apply_spinquant(
            model,
            use_r1=True,
            use_r2=True,
            use_r4=False,
            pretrained_rotation_path=None,
            qkv_split=True,
        )
        logging.info("Applied SpinQuant to the model")

    if args.range_setting == "mse_with_act_loss":
        wrapped_model = WrappedLlamaModel(
            model, *atten_mask, args.use_kv_cache, args.max_seq_length, args.device
        )
        act_bits, weight_bits = {
            "8a8w": (8, 8),
            "16a4w": (16, 4),
            "16a4w_block": (16, 4),
        }[args.ptq]
        scales_state_dict = compute_scales(
            wrapped_model, tokens, weight_bits, act_bits, 1600
        )
        torch.save(scales_state_dict, "scales_state_dict.pth")
        logging.info("Saved scales to scales_state_dict.pth")
        reverse_quantize_module_swap(wrapped_model)

    for layer in model.layers:
        if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
            layer.feed_forward.prepare_feedfoward_conv()
    if args.embedding_quantize:
        model = get_quant_embedding_transform(
            embedding_quantize=args.embedding_quantize
        )(model)

    model = convert_linear_to_conv2d(model)
    return model, prefill_config, inputs, scales_state_dict


def run_static_prefill_lm_eval(
    module,
    get_example_inputs,
    tokenizer,
    max_seq_length: int,
    tasks,
    use_i64_token: bool,
    num_fewshot=None,
    limit=None,
    event_name: str = "",
) -> None:
    wrapper = StaticPrefillEvalWrapper(
        model=module,
        tokenizer=tokenizer,
        get_example_inputs=get_example_inputs,
        max_seq_length=max_seq_length,
        use_i64_token=use_i64_token,
        device=next(module.parameters()).device.type,
    )
    with torch.no_grad():
        eval_results = simple_evaluate(
            model=wrapper,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
        )
    logging.info("Evaluation summary for %s", event_name)
    for task, res in eval_results["results"].items():
        logging.info("%s: %s", task, res)


def eval_llm(args) -> None:
    tokenizer = prepare_tokenizer(args)
    model, prefill_config = prepare_model(args)
    model, config, inputs, scales_state_dict = prequant_algorithm(
        model, prefill_config, args
    )
    use_i64_token = args.embedding_quantize is not None

    if args.ptq is not None:
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")
        decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
        custom_annotations = (
            decoder_model_config.quant_recipe().recipe.custom_quant_annotations
        )

        quantizer = make_custom_quantizer(
            quant_dtype,
            custom_annotations,
            args.quant_linear_only,
            backend=QnnExecuTorchBackendType.kHtpBackend,
            soc_model=args.soc_model,
        )

        with torch.no_grad():
            logging.info("Starting export")
            model = torch.export.export(
                model, (inputs[0], *inputs[1]), strict=True
            ).module()
            if quant_dtype == QuantDtype.use_16a4w_block:
                conv_nodes = [n for n in model.graph.nodes if "conv" in n.name]
                block_size_map = {n.name: (1, 64, 1, 1) for n in conv_nodes}
                quantizer.set_block_size_map(block_size_map)
            logging.info("Finished export, adding observers")
            model = prepare_pt2e(model, quantizer)

        logging.info("Observers added, starting calibration")
        run_static_prefill_lm_eval(
            module=model,
            get_example_inputs=lambda: inputs,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            tasks=["wikitext"],
            use_i64_token=use_i64_token,
            limit=1,
            event_name="prepare_pt2e_tasks",
        )

        if args.range_setting == "mse_with_act_loss":
            set_scales(model, scales_state_dict, config.head_dim)

        logging.info("Quantizing the model")
        model = convert_pt2e(model)

    logging.info("Evaluation of QDQ model")
    run_static_prefill_lm_eval(
        module=model,
        get_example_inputs=lambda: inputs,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tasks=["wikitext"],
        use_i64_token=use_i64_token,
        limit=0.1,
        event_name="convert_pt2e_tasks",
    )


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    parser = build_args_parser()
    parser.add_argument(
        "-P",
        "--ptq",
        help=(
            "If specified, do PTQ quantization. Supports 8a8w, 16a4w, "
            "and 16a4w_block."
        ),
        type=str,
    )
    parser.add_argument(
        "--range_setting",
        help=(
            "Choose range setting method for weight quantization. If not specified, "
            "defaults to minmax."
        ),
        type=str,
    )
    parser.add_argument(
        "--spinquant",
        help="Apply SpinQuant (R1+R2) to the model.",
        action="store_true",
    )
    parser.add_argument(
        "--fraction",
        help="The fraction of examples per task. Only use this for testing.",
        type=float,
    )
    parser.add_argument(
        "--quant_linear_only",
        help="Quantize linear layers only.",
        action="store_true",
    )
    parser.add_argument(
        "--decoder_model",
        help=(
            "The llama model to export. Current available options are: "
            f"{SUPPORTED_LLM_MODELS.keys()}"
        ),
        required=True,
    )
    parser.add_argument(
        "-a",
        "--artifact",
        help="Path for generated artifacts and output by this example.",
        default="./eval_llama_qnn",
        type=str,
    )
    parser.add_argument(
        "--r3",
        help="Enable SpinQuant R3 quantization optimization.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    os.makedirs(args.artifact, exist_ok=True)

    args.generate_full_logits = True
    args.max_seq_len = args.max_seq_length
    args.calibration_seq_length = args.max_seq_length

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(args.device)
    if args.use_attention_sink:
        raise RuntimeError("eval_llama_qnn.py supports CI static prefill eval only")

    args.use_kv_cache = False
    args.prefill_ar_len = args.max_seq_length
    eval_llm(args)


if __name__ == "__main__":
    main()
