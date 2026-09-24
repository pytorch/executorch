# fused_quant

`fused_quant` is a backend-neutral quantization frontend for ExecuTorch. It represents a
`dequantize -> aten op -> quantize` triple as a single `torch.ops.fused_quant.*` operator
that carries its quantization parameters inline, so the quantized graph stays explicit and
optimizable instead of being scattered across q/dq pairs.

It is not a delegate, and its operators have no kernels — nothing executes a
`fused_quant` op directly. It produces an edge-dialect graph that a backend can color,
lower, and partition; anything the backend does not claim is turned back into q/dq + ATen
by `DecomposeFusedQuant` and runs on the ordinary portable kernels. `test_compile.py`
shows that decomposition tail written out for the no-backend case.

## Directory structure

```
backends/fused_quant/
├── ops.py                    fused_quant operator schemas, reference and meta kernels
├── ops_utils.py              shape helpers shared by the op definitions
├── graph_utils.py            qparam / constant / ExportedProgram helpers
├── fuse_aten.py              generic dequant -> op -> quant folding
├── pass_base.py              ExportedProgram pass base + IterativePassGroup
├── passes.py                 the backend-neutral pass pipeline built from those
├── decompose_fused_quant.py  reverse lowering back to q/dq + ATen
├── colorer.py                ColorerBase (claims ops for a backend)
├── frontend.py               trace / prepare / convert
├── compile.py                compile_to_fused_quant, the backend seam
├── quantizer/                torchao PT2E quantizer that also fuses
├── pre_quantize_passes/      rewrites applied before annotation
├── optimization_passes/      rewrites applied to the fused_quant edge graph
└── test/
```

## Operator representation

Every `fused_quant` op takes its tensor operands first, then one qparam block
(`scale`, `zero_point`, `dtype`, `quant_min`, `quant_max`) per quantizable edge, including
the output. A qparam block whose `scale` is `None` means that edge is left in float, so a
single schema covers fully quantized, partially quantized, and float instances of an op.

Granularity is carried by the shape of the `scale` tensor rather than by separate op
variants. The qparams are always shaped to broadcast against the tensor they describe:
per-tensor is a singleton tensor (`scale.numel() == 1`), per-channel has exactly one
non-unary dimension sitting on the quantized axis, and per-group and blockwise tile the
tensor by the ratio of the two shapes (`tensor.shape[i] // scale.shape[i]` per dimension).

## Quantizer

`quantizer/quantizer.py` defines a torchao `Quantizer` that both annotates and fuses:

- `QuantizerBase` — the interface: annotate nodes, then fuse the annotated ones.
- `OpQuantizer` — annotates one ATen op type and folds matched instances into a
  `fused_quant` op. Takes a `QuantizationConfig` plus the schema names of its quantized
  activations, weights, and passthrough operands. Most quantizers are instances of this
  rather than subclasses; subclass it only when an op needs custom fusion.
- `NoopQuantizer` — claims nodes with an empty annotation so a later quantizer skips them,
  leaving the op in float.
- `FusedQuantQuantizer` — composes the above. Annotation is first-match-wins, so a narrower
  quantizer must be listed before the general one it overrides.

`preserved_ops()` reports the ops the quantizer matches so the caller can keep them intact
through `run_decompositions`.

## Known gaps

- The package depends on `torchao`, which is not yet a core ExecuTorch dependency.
- Several optimization passes are shared with the Cadence backend and are still imported
  from `backends/cadence/aot`.
