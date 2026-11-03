# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower the ``hf_static`` float-cache contract to QNN."""

from typing import Any, Optional, Sequence

import torch
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.utils import kv_cache_io
from executorch.backends.qualcomm.utils.export_plan import share_encodings
from executorch.backends.qualcomm.utils.hf_export import DEFAULT_CALIBRATION_PROMPTS
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.llm.export.hf_static import (
    CPU_META,
    DECODE_METHOD,
    flat_call,
    HfStaticLayout,
    IO_CONTRACT,
    IO_CONTRACT_VERSION,
    PREFILL_METHOD,
    sliding_windows,
    specialize,
    tag_cache_mutations,
    to_hf_static,
)
from executorch.extension.llm.io_contracts.hf_static import HfStaticDriver
from torch.export import ExportedProgram

try:
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
except ImportError:
    convert_pt2e = prepare_pt2e = None

try:
    from transformers import GenerationConfig
    from transformers.exporters import DynamoConfig, DynamoExporter
except ImportError:
    DynamoExporter = None


def _cpu_nodes(program):
    """Read explicit rewrite tags, including producers used only by CPU nodes."""
    if isinstance(program, ExportedProgram):
        tag_cache_mutations(program)
    cpu = {n for n in program.graph.nodes if n.meta.get("custom", {}).get(CPU_META)}
    for node in reversed(list(program.graph.nodes)):
        if node.users and all(user in cpu for user in node.users):
            cpu.add(node)
    return [n.name for n in program.graph.nodes if n in cpu]


class HfStaticPartitioner(QnnPartitioner):
    """Keep tagged cache writes, masks and counter arithmetic outside QNN."""

    def transform_passes(self, exported_program: ExportedProgram):
        self.skip_node_id_set |= set(_cpu_nodes(exported_program))
        return super().transform_passes(exported_program)

    def partition(self, edge_program: ExportedProgram):
        self.skip_node_id_set |= set(_cpu_nodes(edge_program))
        return super().partition(edge_program)


def _quantize(
    programs, lengths, layout, quantizer, tokenizer, prompts, seq_mse_candidates=0
):
    """Quantize model compute with float caches and shared weight encodings."""
    if prepare_pt2e is None:
        raise ImportError("hf_static quantization requires torchao PT2E.")
    prepared = {m: prepare_pt2e(p.module(), quantizer) for m, p in programs.items()}
    steps = {m: flat_call(programs[m], prepared[m]) for m in programs}
    decode = HfStaticDriver({1: steps[DECODE_METHOD]}, layout)
    for prompt in prompts:
        ids = tokenizer(prompt)["input_ids"]
        if len(ids) >= layout.max_seq_len:
            raise ValueError(
                f"Calibration prompt of {len(ids)} tokens doesn't fit the cache."
            )
        sequence = decode.generate(ids, layout.max_seq_len - len(ids) - 1)
        if PREFILL_METHOD in prepared:
            chunk = lengths[PREFILL_METHOD]
            prefill = HfStaticDriver(
                {chunk: steps[PREFILL_METHOD], 1: steps[DECODE_METHOD]}, layout
            )
            prefill.run(sequence)
    if seq_mse_candidates:
        first = tokenizer(prompts[0])["input_ids"]
        scales = {}

        def run_decode():
            decode.generate(first, layout.max_seq_len - len(first) - 1)

        kv_cache_io.seq_mse(
            prepared[DECODE_METHOD], seq_mse_candidates, run_decode, scales
        )
        if PREFILL_METHOD in prepared:
            kv_cache_io.seq_mse(
                prepared[PREFILL_METHOD], seq_mse_candidates, lambda: None, scales
            )
    quantized = {}
    for method, program in programs.items():
        args, kwargs = program.example_inputs
        with torch.fx.traceback.preserve_node_meta():
            quantized[method] = torch.export.export(
                convert_pt2e(prepared[method], fold_quantize=True),
                args,
                kwargs,
                strict=False,
            )
        quantized[method].graph_module.meta.update(program.graph_module.meta)
    for method in quantized:
        if method != DECODE_METHOD:
            share_encodings(quantized[DECODE_METHOD], quantized[method], io_only=True)
    return quantized


def _attention_form(attention: str, query_len: int, quantized: bool) -> str:
    """Auto uses split for decode or quantized compute, concat for float prefill."""
    if attention != "auto":
        return attention
    return "split" if query_len == 1 or quantized else "concat"


def _lower(
    hf_graph,
    lengths,
    max_seq_len,
    attention,
    masked_softmax,
    quantizer,
    tokenizer,
    calibration_prompts,
    soc_model,
    partitioner_options,
    metadata,
    seq_mse_candidates=0,
    *,
    config,
):
    forms = {
        m: _attention_form(attention, q, quantizer is not None)
        for m, q in lengths.items()
    }
    programs = {
        m: to_hf_static(
            specialize(hf_graph, q, max_seq_len, config=config),
            masked_softmax,
            forms[m],
        )
        for m, q in lengths.items()
    }
    layout = HfStaticLayout.of(programs[DECODE_METHOD])
    if layout.dtype not in (torch.float16, torch.float32):
        raise ValueError("hf_static caches require float16 or float32.")
    if quantizer is not None:
        programs = _quantize(
            programs,
            lengths,
            layout,
            quantizer,
            tokenizer,
            calibration_prompts,
            seq_mse_candidates,
        )
    options = dict(partitioner_options)
    options.setdefault("use_fp16", quantizer is None)
    options.setdefault("shared_buffer", True)
    options.setdefault("convert_linear_to_conv2d", True)
    options.setdefault("use_weight_sharing", len(programs) > 1)
    if "concat" in forms.values():
        options.setdefault("use_mha2sha", True)
    partitioners = {
        m: [HfStaticPartitioner.from_options(soc_model=soc_model, **options)]
        for m in programs
    }
    constant_methods = {
        **metadata,
        "get_io_contract": IO_CONTRACT,
        "get_io_contract_version": IO_CONTRACT_VERSION,
        "get_max_seq_len": max_seq_len,
        "get_n_layers": layout.n_layers,
        "get_cache_dtype": str(layout.dtype).removeprefix("torch."),
        "use_kv_cache": True,
    }
    edge = to_edge_transform_and_lower(
        programs,
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods={k: v for k, v in constant_methods.items() if v is not None},
    )
    return edge, programs, layout


@torch.no_grad()
def export_hf_static(
    model: torch.nn.Module,
    tokenizer: Any,
    soc_model: str,
    *,
    max_seq_len: int = 128,
    prefill_len: Optional[int] = 32,
    quantizer: Any = None,
    calibration_prompts: Optional[Sequence[str]] = None,
    masked_softmax: Optional[bool] = None,
    attention: Optional[str] = None,
    seq_mse_candidates: int = 0,
    **partitioner_options: Any,
):
    """Export QNN compute with caller-owned float caches and fixed prefill/decode.

    ``auto`` attention uses split for decode and quantized compute, concat for
    float prefill. Prefill must fit every sliding window. The registered runner
    pads prompt chunks and falls back to decode for a near-context tail.
    """
    calibration_prompts = calibration_prompts or DEFAULT_CALIBRATION_PROMPTS
    if (
        not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or max_seq_len <= 0
    ):
        raise ValueError("max_seq_len must be a positive integer.")
    if prefill_len is not None and (
        not isinstance(prefill_len, int)
        or isinstance(prefill_len, bool)
        or not 1 < prefill_len <= max_seq_len
    ):
        raise ValueError(
            "prefill_len must be None or an integer between 2 and max_seq_len."
        )
    windows = sliding_windows(model.config)
    minimum = min([max_seq_len, *[w for w in windows if w is not None]])
    if prefill_len is not None and prefill_len > minimum:
        raise ValueError(
            f"prefill_len ({prefill_len}) exceeds the smallest sliding cache length ({minimum})."
        )
    attention = attention or "auto"
    if attention not in ("split", "concat", "auto"):
        raise ValueError(f"Unknown attention form {attention!r}.")
    masked_softmax = quantizer is not None if masked_softmax is None else masked_softmax
    if DynamoExporter is None:
        raise ImportError("hf_static export requires transformers.exporters.")
    sample = tokenizer(calibration_prompts[0], return_tensors="pt")["input_ids"][
        :, : min(8, minimum)
    ]
    artifacts = DynamoExporter().export_for_generation(
        model.eval(),
        {"input_ids": sample, "attention_mask": torch.ones_like(sample)},
        DynamoConfig(dynamic=True),
        generation_config=GenerationConfig(
            do_sample=False, cache_implementation="static", max_cache_len=max_seq_len
        ),
        keep_all_logits=True,
    )
    lengths = {DECODE_METHOD: 1}
    if prefill_len is not None:
        lengths[PREFILL_METHOD] = prefill_len
    config = model.generation_config
    eos = config.eos_token_id
    metadata = {
        "get_vocab_size": model.config.vocab_size,
        "get_bos_id": config.bos_token_id,
        "get_eos_ids": eos if isinstance(eos, list) or eos is None else [eos],
    }
    edge, _, _ = _lower(
        artifacts.components["decode"].artifact,
        lengths,
        max_seq_len,
        attention,
        masked_softmax,
        quantizer,
        tokenizer,
        calibration_prompts,
        soc_model,
        partitioner_options,
        metadata,
        seq_mse_candidates,
        config=model.config,
    )
    return edge.to_executorch(
        ExecutorchBackendConfig(
            run_reinplace_pass=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
        )
    )
