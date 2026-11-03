# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a generic exporter needs from the QNN backend, in one object.

An exporter (Hugging Face's, or any other) traces a model, optionally quantizes
it with a PT2E quantizer, and lowers it with ``to_edge_transform_and_lower``.
``QnnExportPlan`` fills in the QNN parts of that flow without the exporter
knowing about QNN:

- ``program_transform``: after tracing and before quantization, a program with
  a KV cache is rewritten into the KV-cache-as-I/O contract HTP needs
  (``kv_cache_io``); anything else is left alone.
- ``calibration``: for a rewritten decoder, calibration runs the runner's token
  loop over the calibration prompts; otherwise the exporter's own loop is used.
- ``partitioners``, ``method_name``, ``constant_methods``, ``backend_config``,
  ``edge_compile_config``: the lowering settings, which depend on what the
  transform produced.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import torch
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.utils import kv_cache_io
from executorch.exir import EdgeCompileConfig
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torch.export import ExportedProgram


class QnnExportPlan:
    """QNN's part of a generic trace → quantize → lower export.

    Args:
        soc_model: Target SoC, e.g. ``"SM8750"``.
        backend: QNN backend, ``"htp"`` by default.
        quantizer: The PT2E quantizer the program will be quantized with, or
            None for an fp16 program.
        metadata: Constant methods to carry for a decoder (``get_bos_id``,
            ``get_eos_ids``), as a runtime reads them.
        combine_methods: Lower a decoder's prompt and decode graphs into one
            program (``prefill_forward`` + ``kv_forward``) that shares weights in
            one QNN context, as ``qnn_llama_runner``'s hybrid mode loads it.
        r3: For a decoder, rotate queries and keys by a Hadamard matrix after
            RoPE (see ``to_kv_cache_io``).
        seq_mse_candidates: For a quantized decoder, search each convolution's
            weight scale over this many candidates after calibration (SeqMSE);
            0 skips it.
        seq_mse_steps: A dict shared by the plans of graphs that share weights
            (a decoder's prompt and decode steps): the first fills it, the
            others reuse its weight scales.
        masked_softmax: Mask attention with HTP's MaskedSoftmax form (see
            ``to_kv_cache_io``). Defaults to on for quantized programs, where it
            holds for any score range; an additive mask is faster on models
            whose scores stay small.
        **options: Further ``QnnPartitioner.from_options`` options.
    """

    def __init__(
        self,
        soc_model: str,
        backend: str = "htp",
        quantizer: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        combine_methods: bool = True,
        r3: bool = False,
        seq_mse_candidates: int = 0,
        masked_softmax: Optional[bool] = None,
        seq_mse_steps: Optional[Dict[str, float]] = None,
        **options: Any,
    ) -> None:
        self.soc_model = soc_model
        self.backend = backend
        self.quantizer = quantizer
        self.metadata = dict(metadata or {})
        self.options = dict(options)
        if backend == "htp":
            self.options.setdefault("use_fp16", quantizer is None)
        self.kv_io: Optional[kv_cache_io.KvCacheIoProgram] = None
        self.method_name = "forward"
        self.combine_methods = combine_methods
        self.r3 = r3
        self.seq_mse_candidates = seq_mse_candidates
        self.seq_mse_steps = seq_mse_steps
        self.masked_softmax = (
            quantizer is not None if masked_softmax is None else masked_softmax
        )

    # Trace → transform → quantize.
    def program_transform(self, exported_program: ExportedProgram) -> ExportedProgram:
        """Rewrite a program with a KV cache into the KV-cache-as-I/O contract."""
        if not kv_cache_io.has_kv_cache(exported_program):
            return exported_program
        self.kv_io = kv_cache_io.to_kv_cache_io(
            exported_program,
            self.metadata,
            masked_softmax=self.masked_softmax,
            r3=self.r3,
        )
        self.method_name = self.kv_io.method_name
        return self.kv_io.export()

    def calibration(
        self, dataset: Optional[Iterable[Any]]
    ) -> Optional[Callable[[torch.nn.Module], None]]:
        """The token-loop calibration for a rewritten decoder, or None to let
        the exporter calibrate as usual. ``dataset`` holds prompts as token-id
        lists or as dicts with ``input_ids`` (and, for an encoder-decoder,
        ``cross_kv``: the step's cross-attention inputs for that sample)."""
        if self.kv_io is None or not dataset:
            return None
        prompts, cross_kv = [], []
        for sample in dataset:
            ids = sample.get("input_ids") if isinstance(sample, dict) else sample
            if isinstance(ids, torch.Tensor):
                ids = ids.reshape(-1).tolist()
            prompts.append(list(ids))
            if isinstance(sample, dict) and "cross_kv" in sample:
                cross_kv.append(sample["cross_kv"])
        eos = self.metadata.get("get_eos_ids") or []
        return self.kv_io.calibration_fn(
            prompts,
            eos if isinstance(eos, list) else [eos],
            seq_mse_candidates=self.seq_mse_candidates,
            cross_kv=cross_kv or None,
            seq_mse_steps=self.seq_mse_steps,
        )

    # Lowering.
    def partitioners(self) -> List[QnnPartitioner]:
        options = dict(self.options)
        if self.kv_io is not None:
            # HTP's decoder settings: multi-head attention split per head, and
            # linear layers run as 1x1 convolutions.
            options.setdefault("use_mha2sha", True)
            options.setdefault("convert_linear_to_conv2d", True)
            # The runner's KV buffers are shared (rpcmem) memory, and a prompt
            # and a decode step compiled together share weights.
            options.setdefault("shared_buffer", True)
            options.setdefault("use_weight_sharing", True)
            if self.quantizer is not None:
                options.setdefault("passes_job", self._quant_io_passes())
        return [
            QnnPartitioner.from_options(
                soc_model=self.soc_model, backend=self.backend, **options
            )
        ]

    def group(self) -> Optional[str]:
        """Graphs of one decoder are lowered together, so they share weights."""
        return "decoder" if self.kv_io is not None and self.combine_methods else None

    def merge(self, programs: Dict[str, ExportedProgram]) -> Dict[str, ExportedProgram]:
        """Give every graph of the decoder the decode step's quantization
        encodings: the runner hands the same KV buffers to all of them, and
        weight sharing needs identical weight encodings."""
        if self.quantizer is None or kv_cache_io.METHOD_NAME not in programs:
            return programs
        source = programs[kv_cache_io.METHOD_NAME]
        for name, program in programs.items():
            if name != kv_cache_io.METHOD_NAME:
                share_encodings(source, program)
        return programs

    def constant_methods(
        self, program: Optional[ExportedProgram] = None
    ) -> Optional[Dict[str, Any]]:
        if self.kv_io is None:
            return None
        methods = self.kv_io.constant_methods()
        if program is not None and self.quantizer is not None:
            # The logits' quantization, so a runtime can dequantize them.
            node = next(n for n in program.graph.nodes if n.op == "output").args[0][0]
            while node.op == "call_function" and node.target not in _PER_TENSOR_QDQ:
                node = node.args[0]
            if node.op == "call_function":
                methods["get_logits_scale"] = float(node.args[1])
                methods["get_logits_zero_point"] = int(node.args[2])
        return methods

    def edge_compile_config(self) -> EdgeCompileConfig:
        # QNN's own passes emit non-core ATen ops.
        return EdgeCompileConfig(_check_ir_validity=False)

    def backend_config(
        self, alloc_graph_input: bool = True, alloc_graph_output: bool = True
    ):
        from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo

        if self.kv_io is not None:
            # The runtime binds the cache buffers itself.
            alloc_graph_input = alloc_graph_output = False
        return ExecutorchBackendConfig(
            passes=[BuildQuantIo()],
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=alloc_graph_input,
                alloc_graph_output=alloc_graph_output,
            ),
        )

    def _quant_io_passes(self):
        from executorch.backends.qualcomm._passes import TagQuantIO
        from executorch.backends.qualcomm._passes.qnn_pass_manager import (
            get_qnn_pass_manager_cls,
        )
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_PASS_ACTIVATE_KEY,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
        )

        bits = _activation_bits(self.quantizer)
        kv_dtype = torch.uint8 if bits == 8 else torch.uint16
        passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()
        passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = self.kv_io.quant_io_dtype_fn(kv_dtype, torch.uint16)
        return passes_job


def _activation_bits(quantizer) -> int:
    """Activation width of the quantizer's default config: the KV cache is an
    activation, so this is the width it crosses the graph boundary at."""
    from executorch.backends.qualcomm.quantizer.custom_annotation import (
        annotate_kv_8bit,
    )
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

    recipe = getattr(quantizer, "_recipe", None)
    if recipe is not None:
        if annotate_kv_8bit in (
            getattr(c, "func", c) for c in recipe.custom_quant_annotations
        ):
            return 8
        dtype = recipe._default_quant_dtype
    else:
        dtype = quantizer.default_quant_config.quant_dtype
    return 8 if dtype in (QuantDtype.use_8a8w, QuantDtype.use_8a4w) else 16


_PER_TENSOR_QDQ = (
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
)


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
