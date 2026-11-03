# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a Hugging Face decoder LLM to a QNN ``.pte`` from ExecuTorch.

Hugging Face exporters only trace: ``DynamoExporter.export_for_generation``
with a static cache gives each generation step as a program that takes its
cache as inputs. Everything after that is ExecuTorch's and QNN's, through the
standard APIs: the KV-cache-as-I/O rewrite (``to_kv_cache_io``), PT2E
quantization with QNN's token-loop calibration, and
``to_edge_transform_and_lower``. By default the program is labelled
``kvcache_io``, which ExecuTorch's LLM runner (``KvCacheIOManager``) and
``qnn_llama_runner`` drive; with ``--io_contract hf_static`` it keeps Hugging
Face's static-cache ABI (``hf_static``), which ExecuTorch's LLM runner
drives through ``HfStaticIOManager``.

    python -m executorch.backends.qualcomm.utils.hf_export \\
        --model HuggingFaceTB/SmolLM2-135M-Instruct --soc_model SM8750 \\
        --quantize --hybrid --output smollm2.pte
"""

import argparse
import copy
from typing import Any, Dict, Optional, Sequence

import torch
from executorch.backends.qualcomm.utils.export_plan import QnnExportPlan
from executorch.exir import ExecutorchProgramManager, to_edge_transform_and_lower

DEFAULT_CALIBRATION_PROMPTS = (
    "Once upon a time",
    "The capital of France is",
    "In a shocking finding, scientists discovered",
    "The history of the Roman Empire began when",
    "To make a good cup of coffee, you need",
    "The stock market fell sharply on Monday after",
    "Photosynthesis is the process by which plants",
    "def fibonacci(n):",
)


def _quantize(program, quantizer, calibrate):
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    args, kwargs = program.example_inputs
    prepared = prepare_pt2e(program.module(), quantizer)
    calibrate(prepared)
    converted = convert_pt2e(prepared, fold_quantize=True)
    return torch.export.export(
        converted, copy.deepcopy(args), copy.deepcopy(kwargs), strict=False
    )


def export_hf_decoder(
    model: torch.nn.Module,
    tokenizer: Any,
    soc_model: str,
    *,
    max_seq_len: int = 128,
    prefill_len: Optional[int] = None,
    quantizer: Any = None,
    calibration_prompts: Sequence[str] = DEFAULT_CALIBRATION_PROMPTS,
    io_contract: str = "kvcache_io",
    **plan_options: Any,
) -> ExecutorchProgramManager:
    """Export a decoder LLM to one QNN program.

    Args:
        model: A Hugging Face causal LM.
        tokenizer: Its tokenizer, for the sample and calibration prompts.
        soc_model: Target SoC, e.g. ``"SM8750"``.
        max_seq_len: Context length: the static cache's size.
        prefill_len: With a length, the program also gets a prompt graph
            taking that many tokens per step, sharing weights with the decode
            graph.
        quantizer: A PT2E QNN quantizer, or None for fp16.
        calibration_prompts: Prompts QNN's calibration generates from.
        io_contract: The program's cache contract. ``"kvcache_io"``: QNN's
            KV cache as I/O (``prefill_forward``/``kv_forward``; the runner
            writes the new rows). ``"hf_static"``: Hugging Face's
            static-cache ABI unchanged (``prefill``/``decode``; the program writes
            the new rows into the caller's caches, see ``hf_static``).
        **plan_options: For ``kvcache_io``, ``QnnExportPlan`` options (``r3``,
            ``seq_mse_candidates``, ``masked_softmax``, partitioner options);
            for ``hf_static``, ``masked_softmax``, ``seq_mse_candidates``
            and partitioner options (not ``r3``:
            it would change what the caches hold).
    """
    if io_contract == "hf_static":
        from executorch.backends.qualcomm.utils import hf_static

        if plan_options.pop("r3", False):
            # R3 would change what the caches hold, which HF's ABI exposes.
            raise ValueError("r3 is not supported with hf_static.")
        return hf_static.export_hf_static(
            model,
            tokenizer,
            soc_model,
            max_seq_len=max_seq_len,
            prefill_len=prefill_len,
            quantizer=quantizer,
            calibration_prompts=calibration_prompts,
            **plan_options,
        )
    if io_contract != "kvcache_io":
        raise ValueError(f"Unknown io_contract {io_contract!r}.")
    from transformers import GenerationConfig
    from transformers.exporters import DynamoConfig, DynamoExporter

    if prefill_len:
        words = tokenizer(
            " ".join(["word"] * 2 * prefill_len),
            return_tensors="pt",
            add_special_tokens=False,
        )
        sample = words["input_ids"][:, :prefill_len]
    else:
        # A short sample: the static cache's length must not grow to fit it.
        sample = tokenizer(calibration_prompts[0], return_tensors="pt")["input_ids"]
        sample = sample[:, : min(sample.shape[1], 8)]
    generation = GenerationConfig(
        do_sample=False, cache_implementation="static", max_cache_len=max_seq_len
    )
    artifacts = DynamoExporter().export_for_generation(
        model.eval(),
        {"input_ids": sample, "attention_mask": torch.ones_like(sample)},
        DynamoConfig(),
        generation_config=generation,
    )

    config = model.generation_config
    eos = config.eos_token_id
    metadata = {
        "get_bos_id": config.bos_token_id,
        "get_eos_ids": eos if isinstance(eos, list) or eos is None else [eos],
    }
    prompts = [tokenizer(p)["input_ids"] for p in calibration_prompts]
    plans: Dict[str, QnnExportPlan] = {}
    programs = {}
    # The decode step searches weight scales once; the prompt step reuses them.
    plan_options.setdefault("seq_mse_steps", {})
    for component in ("decode", "prefill") if prefill_len else ("decode",):
        plan = QnnExportPlan(
            soc_model, quantizer=quantizer, metadata=metadata, **plan_options
        )
        program = plan.program_transform(artifacts.components[component].artifact)
        if quantizer is not None:
            program = _quantize(program, quantizer, plan.calibration(prompts))
        plans[plan.method_name] = plan
        programs[plan.method_name] = program

    decode = plans["kv_forward"]
    programs = decode.merge(programs)
    constant_methods = {}
    for method, plan in plans.items():
        constant_methods.update(plan.constant_methods(programs[method]) or {})
    edge = to_edge_transform_and_lower(
        programs,
        partitioner={method: plan.partitioners() for method, plan in plans.items()},
        compile_config=decode.edge_compile_config(),
        constant_methods=constant_methods,
    )
    return edge.to_executorch(config=decode.backend_config())


class _CrossAttentionInputs(torch.nn.Module):
    """An encoder-decoder's encoder, plus the cross-attention keys (transposed)
    and values each decoder layer computes from its output: the fixed inputs
    of the rewritten decode step, computed once per input."""

    def __init__(self, model: torch.nn.Module, start_token: int):
        super().__init__()
        self.model = model
        self.start_token = start_token

    def forward(self, features: torch.Tensor):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        states = self.model.get_encoder()(features)[0]
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        self.model.get_decoder()(
            input_ids=torch.full((1, 1), self.start_token, dtype=torch.long),
            encoder_hidden_states=states,
            past_key_values=cache,
            use_cache=True,
        )
        layers = cache.cross_attention_cache.layers
        return (
            *[layer.keys.transpose(2, 3).contiguous() for layer in layers],
            *[layer.values for layer in layers],
        )


def export_hf_seq2seq(
    model: torch.nn.Module,
    soc_model: str,
    features: Sequence[torch.Tensor],
    *,
    max_seq_len: int = 128,
    quantizer: Any = None,
    encoder_quantizer: Any = None,
    **plan_options: Any,
) -> ExecutorchProgramManager:
    """Export an encoder-decoder model (e.g. Whisper) to one QNN program.

    The program has an ``encoder`` method, returning the cross-attention keys
    and values, and a ``kv_forward`` decode step that takes them as fixed
    inputs after its self-attention caches (``kvcache_io`` with
    ``get_n_cross_layers``).

    Args:
        model: A Hugging Face encoder-decoder model.
        soc_model: Target SoC, e.g. ``"SM8750"``.
        features: Encoder inputs (e.g. log-mel features): the first is the
            sample to trace with, all calibrate.
        max_seq_len: Decoder context length: the static self-attention cache.
        quantizer: A PT2E QNN quantizer for the decoder, or None for fp16.
        encoder_quantizer: The encoder's quantizer (defaults to ``quantizer``).
        **plan_options: ``QnnExportPlan`` options.
    """
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from transformers.exporters import DynamoConfig, DynamoExporter

    model = model.eval()
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.cache_implementation = "static"
    generation.max_cache_len = max_seq_len
    generation.max_new_tokens = max_seq_len // 2
    artifacts = DynamoExporter().export_for_generation(
        model,
        {"input_features": features[0]},
        DynamoConfig(),
        generation_config=generation,
    )
    start = model.config.decoder_start_token_id
    eos = generation.eos_token_id
    metadata = {
        "get_bos_id": start,
        "get_eos_ids": eos if isinstance(eos, list) or eos is None else [eos],
    }
    plan = QnnExportPlan(
        soc_model, quantizer=quantizer, metadata=metadata, **plan_options
    )
    decode = plan.program_transform(artifacts.components["decode"].artifact)

    cross = _CrossAttentionInputs(model, start)
    encoder = torch.export.export(cross, (features[0],), strict=False)
    if quantizer is not None:
        with torch.no_grad():
            samples = [cross(f) for f in features]
        dataset = [{"input_ids": [start], "cross_kv": kv} for kv in samples]
        decode = _quantize(decode, quantizer, plan.calibration(dataset))
        prepared = prepare_pt2e(encoder.module(), encoder_quantizer or quantizer)
        for f in features:
            prepared(f)
        encoder = torch.export.export(
            convert_pt2e(prepared, fold_quantize=True), (features[0],), strict=False
        )
    edge = to_edge_transform_and_lower(
        {"encoder": encoder, "kv_forward": decode},
        partitioner={
            "encoder": [
                QnnPartitioner.from_options(
                    soc_model=soc_model,
                    use_fp16=quantizer is None,
                    shared_buffer=True,
                )
            ],
            "kv_forward": plan.partitioners(),
        },
        compile_config=plan.edge_compile_config(),
        constant_methods=plan.constant_methods(decode),
    )
    return edge.to_executorch(config=plan.backend_config())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Hugging Face model id or path")
    parser.add_argument("--soc_model", default="SM8750")
    parser.add_argument("--output", required=True, help="Path of the .pte")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Add a prompt graph (prefill_forward) next to the decode graph",
    )
    parser.add_argument("--prefill_len", type=int, default=32)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize with the model's tuned recipe (16-bit activations by default)",
    )
    parser.add_argument(
        "--recipe_id", default=None, help="Recipe to use instead of the model's"
    )
    parser.add_argument(
        "--io_contract",
        choices=("kvcache_io", "hf_static"),
        default="kvcache_io",
        help="The program's cache contract (see export_hf_decoder)",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    quantizer, plan_options = None, {}
    if args.quantize:
        from executorch.backends.qualcomm.export_utils import (
            get_backend_type,
            make_quantizer,
        )
        from executorch.backends.qualcomm.quantizer.hf_llm_quant_recipe import (
            get_quant_recipe,
        )

        recipe = get_quant_recipe(args.recipe_id or args.model)
        quantizer = make_quantizer(
            backend=get_backend_type("htp"), soc_model=args.soc_model
        )
        quantizer.set_recipe(recipe.recipe)
        quantizer.set_convert_linear_to_conv2d(True)
        plan_options = {
            "masked_softmax": recipe.masked_softmax,
            "seq_mse_candidates": recipe.seq_mse_candidates,
        }
        if args.io_contract == "kvcache_io":
            plan_options["r3"] = recipe.r3
    program = export_hf_decoder(
        model,
        tokenizer,
        args.soc_model,
        max_seq_len=args.max_seq_len,
        prefill_len=args.prefill_len if args.hybrid else None,
        quantizer=quantizer,
        io_contract=args.io_contract,
        **plan_options,
    )
    with open(args.output, "wb") as f:
        program.write_to_file(f)


if __name__ == "__main__":
    main()
