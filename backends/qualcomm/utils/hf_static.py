# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower the ``hf_static`` float-cache contract to QNN."""

import argparse
import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import torch
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
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
_SEQ_MSE_STEPS = 32
_R3_WEIGHT = "_qnn_r3"
_PER_TENSOR_QDQ = (
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
)


def seq_mse(
    module: torch.fx.GraphModule,
    num_candidates: int,
    run: Callable[[], None],
    steps: Optional[Dict[str, float]] = None,
) -> None:
    """SeqMSE over every forward ``run`` makes: QNN's SeqMSE searches on each
    forward it sees and keeps the last result, so with one token per step the
    inputs are gathered first and searched once.

    ``steps`` maps each weight to its chosen scale step: filled when empty,
    applied without searching otherwise, so graphs sharing weights (a prompt
    and a decode step) quantize them identically."""
    from executorch.backends.qualcomm._passes.seq_mse import (
        InsertSeqMse,
        RemoveSeqMse,
        SeqMseModule,
    )

    # The rewritten program holds its weights as constants, not parameters.
    module.get_parameter = lambda target: functools.reduce(
        getattr, target.split("."), module
    )
    try:
        InsertSeqMse(num_candidates)(module)
    finally:
        del module.get_parameter
    # Each search, keyed by its weight (conv -> weight observer -> weight).
    searches = {
        node.args[1].args[1].args[0].target: getattr(module, node.target)
        for node in module.graph.nodes
        if node.op == "call_module"
        and isinstance(getattr(module, node.target), SeqMseModule)
    }
    # The Hadamard matrix's entries are all +-1/sqrt(d): any smaller scale
    # would clip it.
    searches.pop(_R3_WEIGHT, None)
    if steps:
        for weight, m in searches.items():
            m.best_candidate_step = steps.get(weight, 1.0)
    else:
        seen = {weight: ([], []) for weight in searches}
        for weight, m in searches.items():
            m.forward = lambda x, y, io=seen[weight]: (
                len(io[0]) < _SEQ_MSE_STEPS
                and (io[0].append(x.detach()), io[1].append(y.detach()))
            )
        run()
        for weight, m in searches.items():
            inputs, outputs = seen[weight]
            if inputs:
                m.best_candidate_step = m._find_best_candidate(
                    torch.cat(inputs), torch.cat(outputs)
                )
            if steps is not None:
                steps[weight] = m.best_candidate_step
    RemoveSeqMse()(module)


def _encoding_key(
    node: torch.fx.Node, output_index: Dict[torch.fx.Node, int], inputs: Set[str]
):
    """Where a quantize/dequantize node's encoding comes from, comparable across
    graphs of one model: a graph input by name, a graph output by position,
    anything else by the op and module that produce the value. Constants
    (numbered per graph) are left alone."""
    producer = node.args[0]
    while producer.op == "call_function" and producer.target in _PER_TENSOR_QDQ:
        producer = producer.args[0]
    if producer.op == "placeholder" and producer.name in inputs:
        return ("input", producer.name)
    if producer.op == "placeholder":
        # A constant's encoding follows from its value, the same in every graph.
        return ("constant", producer.name)
    users = [node]
    while users:
        user = users.pop()
        if user in output_index:
            return ("output", output_index[user])
        users.extend(
            u
            for u in user.users
            if u.op == "call_function" and u.target in _PER_TENSOR_QDQ
        )
    stack = producer.meta.get("nn_module_stack") or {}
    path = list(stack.values())[-1][0] if stack else ""
    return (str(producer.target), path)


def share_encodings(
    source: ExportedProgram, target: ExportedProgram, io_only: bool = False
) -> int:
    """Copy the per-tensor activation encodings of ``source`` onto ``target``,
    two quantized graphs of the same model that differ in step length (a decode
    step and a prompt step). Values are matched by where their encoding comes
    from (graph input name, graph output position, or producing op and module,
    in order of appearance); the graph inputs and outputs, which the runtime
    exchanges between the two graphs, must all match. Returns the number of
    encodings copied. With ``io_only``, only the graph inputs' and outputs'
    (graphs whose insides differ, e.g. in their attention form)."""

    def keyed(program):
        out = next(n for n in program.graph.nodes if n.op == "output")
        output_index = {
            arg: i
            for i, arg in enumerate(out.args[0])
            if isinstance(arg, torch.fx.Node)
        }
        inputs = set(program.graph_signature.user_inputs)
        groups: Dict[Any, List[torch.fx.Node]] = {}
        for n in program.graph.nodes:
            if n.op == "call_function" and n.target in _PER_TENSOR_QDQ:
                groups.setdefault(
                    (_encoding_key(n, output_index, inputs), n.target), []
                ).append(n)
        return groups

    src, dst = keyed(source), keyed(target)
    copied = 0
    for key, nodes in dst.items():
        if io_only and key[0][0] not in ("input", "output"):
            continue
        peers = src.get(key)
        if peers is not None and len({tuple(p.args[1:]) for p in peers}) == 1:
            # One encoding however many nodes carry it (e.g. a dequantize per
            # user of a graph input).
            peers = peers[:1] * len(nodes)
        if peers is None or len(peers) != len(nodes):
            if key[0][0] in ("input", "output"):
                raise RuntimeError(
                    f"Cannot share the encoding of graph I/O {key[0]}: no matching node."
                )
            continue
        for a, b in zip(peers, nodes):
            b.args = (b.args[0], *a.args[1:])
            copied += 1
    target.graph_module.recompile()
    return copied


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

        seq_mse(prepared[DECODE_METHOD], seq_mse_candidates, run_decode, scales)
        if PREFILL_METHOD in prepared:
            seq_mse(prepared[PREFILL_METHOD], seq_mse_candidates, lambda: None, scales)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model id or path")
    parser.add_argument("--soc_model", default="SM8750")
    parser.add_argument("--output", required=True, help="Path of the .pte")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--hybrid", action="store_true", help="Add a fixed prefill method"
    )
    parser.add_argument("--prefill_len", type=int, default=32)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model computation while keeping float caches",
    )
    parser.add_argument(
        "--recipe_id", default=None, help="Recipe to use instead of the model's"
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    quantizer, options = None, {}
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
        options = {
            "masked_softmax": recipe.masked_softmax,
            "seq_mse_candidates": recipe.seq_mse_candidates,
        }
    program = export_hf_static(
        model,
        tokenizer,
        args.soc_model,
        max_seq_len=args.max_seq_len,
        prefill_len=args.prefill_len if args.hybrid else None,
        quantizer=quantizer,
        **options,
    )
    with open(args.output, "wb") as f:
        program.write_to_file(f)


if __name__ == "__main__":
    main()
