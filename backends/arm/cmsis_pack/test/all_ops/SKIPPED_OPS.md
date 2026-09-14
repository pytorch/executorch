# All-ops test: skipped operators and how to cover them

This test gives **every** operator the pack ships two kinds of coverage:

- **Build / link** — `gen_cproject.py` selects all 198 operator components, so the
  consumer firmware links every op's generated registration, forward declaration and
  kernel source. *No operator is exempt from this.*
- **Execution** — `generate_test_models.py` turns each op's recipe in `op_recipes.py`
  into a `.pte` + reference output that the firmware runs on the FVP and checks.

A **skip** only means an op is not *executed*; it is still build/link-covered. Every skip
carries a reason (enforced by `test_op_recipes.py::test_no_silent_coverage_gaps`), so
there are no silent gaps. Current status on the CPU target (Corstone-315): **167 executed
(incl. 2 shape variants), 33 skipped** of 198 operator components, all passing.

The Ethos-U variant has a separate, weaker status — see
[Ethos-U variant status](#ethos-u-variant-status) before citing its results.

The sections below group the skips by root cause, with a concrete way to close each.

---

## 1. Nondeterministic output — `rand`, `randn`, `native_dropout`

**Why:** the output depends on an RNG draw. Even with a fixed seed, the device runtime's
generator does not reproduce PyTorch's stream, so there is no stable reference to compare
against bit-for-bit.

**Solution:** add a *structural* assertion mode (compare shape + dtype only, not values) to
the manifest and the runner, and run these ops in that mode. For `native_dropout` in eval
mode the op is identity, so it can additionally be value-checked against its input. For
`rand`/`randn`, optionally assert distribution statistics (mean/std within a loose
tolerance) instead of exact values.

**Effort:** small — one `"check": "shape"` field in the manifest + a branch in
`tensors_match`.

---

## 2. Uninitialized contents — `empty`, `empty_dim_order`

**Why:** `empty` allocates without initializing, so the contents are arbitrary by contract.

**Solution:** same structural (shape + dtype only) assertion mode as §1. The kernel still
runs; we just don't compare values.

**Effort:** small (shares the §1 mechanism).

---

## 3. Data-dependent output shape — `nonzero`, `masked_select`

**Why:** the output length depends on the input *values* (how many elements are nonzero /
selected). That makes the output an unbacked dynamic shape, which static memory planning
and a fixed `expected.bin` cannot express.

**Solution:** pin the result size by construction — feed an input whose selected/nonzero
count is a known constant (e.g. a mask with exactly K `True`s) and export with that as a
fixed upper bound. The output shape then becomes constant for that input and the reference
compare works. Requires the ET lean runtime to support the op's dynamic output; if it does
not, exercise the op behind a fixed-size consumer (e.g. `masked_select(...)[:K]`) instead.

**Effort:** medium — depends on lean-runtime dynamic-shape support.

---

## 4. Backward / training-only — `convolution_backward`, `max_pool2d_with_indices_backward`

**Why:** these are gradient ops with no forward-inference form; a normal `nn.Module`
forward never emits them.

**Solution:** call the backward op directly with synthetic tensors that match its schema —
a `grad_output`, the saved `input`/`weight` (or pooling `indices`), and the boolean
output-mask. Export that as the model. Low priority: these almost never run in edge
inference, but the recipe is mechanical once the signature is filled in.

**Effort:** medium, low value.

---

## 5. Creation ops constant-folded at export — `arange`, `full`, `ones`, `zeros`, `scalar_tensor`

**Why:** with constant arguments, `torch.export` + `to_edge` fold these into a frozen
constant tensor baked into the `.pte`. The kernel is never *called* at runtime, so there is
nothing to execute or observe.

**Solution:** make the creation depend on a runtime value so it cannot be folded — e.g.
`torch.arange(x.shape[0])` with a dynamic batch dim, or `torch.full((n,), x[0].item())`
with the fill value taken from an input. This keeps the op live in the graph. Needs
dynamic-shape / data-dependent export. Otherwise these remain legitimately build/link-only
(the kernel ships and links; it is simply not invoked by a static graph).

**Effort:** medium (dynamic-shape export), or accept as build-link-only.

---

## 6. Dim-order no-op for the default layout — `clone_dim_order`, `to_dim_order_copy`

**Why:** with the default contiguous layout these copies are no-ops and get elided, so the
reorder path is never exercised.

**Solution:** build an input with a non-default dim order (e.g. channels-last) and request
the opposite order, so the copy actually permutes memory. Use the ET `dim_order` export
APIs to force the conversion into the graph.

**Effort:** small–medium (need the dim-order export knobs); straightforward recipe.

---

## 7. Just needs a bespoke recipe — `grid_sampler_2d`, `cdist_forward`, `pdist_forward`, `upsample_bilinear2d_aa`

**Why:** no fundamental blocker — they only need specific inputs the bulk recipes don't
provide (a sampling grid in `[-1, 1]`; two point sets for the distance ops; the
`antialias=True` flag).

**Solution:** add explicit recipes:
- `grid_sampler_2d`: `F.grid_sample(input, grid)` with `grid` in `[-1, 1]`.
- `cdist_forward`: `torch.cdist(a, b)` with `a:(B,N,D)`, `b:(B,M,D)`.
- `pdist_forward`: `torch.pdist(x)` with `x:(N,D)`.
- `upsample_bilinear2d_aa`: `F.interpolate(x, scale_factor=2, mode="bilinear", antialias=True)`.

**Effort:** small — a few lines each in `op_recipes.py`.

---

## 8. ET out-variant shape — `allclose`

**Why:** `allclose` returns a scalar bool, but ExecuTorch out-variants must write into a
provided `out` tensor. The pack's custom `allclose.out` writes a single-element bool, so the
recipe needs that exact output shape.

**Solution:** recipe that produces a single-element bool `out` and compares exactly.

**Effort:** small.

---

## 9. Complex dtype unsupported on target — `view_as_real_copy`

**Why:** needs a `complex64` input; complex dtypes are not supported on the Cortex-M
portable-lean target.

**Solution:** if/when the target supports complex, add a complex-input recipe; until then
this is genuinely out of scope and stays build/link-only.

**Effort:** blocked on target support.

---

## 10. `quantized_decomposed::*` ops — `add`, `choose_qparams`, `dequantize`, `quantize`, `embedding`, `embedding2b`, `embedding4b`, `mixed_linear`, `mixed_mm`

**Why:** these kernels (`kernels/quantized/`) are only invoked when the graph contains
`quantized_decomposed.*` ops, which the **PT2E quantization flow** emits. A plain float
portable export never produces them, so the float recipe path cannot reach them.

**Solution:** drive them through quantization rather than a float model:
- For `quantize`/`dequantize`/`choose_qparams`/`add`: run `prepare_pt2e` → `convert_pt2e`
  with a quantizer that emits `quantize_per_tensor`/`dequantize_per_tensor`, or call the
  `torch.ops.quantized_decomposed.*` ops directly with explicit `scale`/`zero_point`.
- For `embedding_byte` / `2bit` / `4bit`: construct the packed int weight + per-channel
  scales and call `torch.ops.quantized_decomposed.embedding_*`.
- For `mixed_linear` / `mixed_mm`: build the mixed-dtype operand pair the op expects.

A `_export_quantized` path (sibling to `_export_portable` / `_export_cortex_m`) in
`generate_test_models.py` would host this. Wire it via `exporters["Quantized"]`.

**Effort:** medium — one quantized export path + 9 recipes with explicit quant params.

---

## Not a skip: Cortex-M ops are environment-gated — 16 ops

`quantized_add`, `quantized_mul`, `maximum`, `minimum`, `quantized_linear`, `softmax`,
`transpose`, `pad`, `quantized_conv2d`, `quantized_depthwise_conv2d`,
`quantized_transpose_conv2d`, `quantized_avg_pool2d`, `quantized_max_pool2d`,
`quantized_batch_matmul`, `quantize_per_tensor`, `dequantize_per_tensor`.

These **have recipes** (`_export_cortex_m`, via `CortexMQuantizer` + `CortexMPassManager`).
They fail to export *only* in an environment without the `cmsis_nn` Python dependency — the
Cortex-M backend raises `ModuleNotFoundError: cmsis_nn` at import, identically for all 16,
before any recipe runs.

**Solution:** install the dep and re-run:

```bash
examples/arm/setup.sh --i-agree-to-the-contained-eula   # installs cmsis_nn
# or:  pip install --no-dependencies -r backends/cortex_m/requirements-cortex-m.txt
python backends/arm/cmsis_pack/test/all_ops/generate_test_models.py \
    --source-dir <repo> --output-dir <out> --continue-on-error
```

The Docker image used for the consumer build already provides this, so they export there.

---

## Priority to raise execution coverage

1. **§7 + §8 bespoke recipes** (5 ops) — trivial, immediate wins.
2. **§1 + §2 structural-assertion mode** (5 ops) — one small runner/manifest feature.
3. **Cortex-M env** — install `cmsis_nn`; unlocks 16 ops with no new code.
4. **§10 quantized path** (9 ops) — one quantized export path.
5. **§6 dim-order** (2 ops), **§5 creation** (5 ops), **§3 data-dependent** (2 ops) —
   need dynamic-shape / dim-order export plumbing.
6. **§4 backward** (2), **§9 complex** (1) — lowest value / blocked on target support.

---

## Ethos-U variant status

`ETHOS_U=1` builds the Corstone-320 (SSE-320 + Ethos-U85) variant, which routes every
recipe through Vela: whatever partitions runs on the NPU, the rest falls back to the host
core. It is **build/link and execution coverage only — it does not currently establish
numerical correctness**, and should not be cited as evidence that the NPU path is correct.

Measured on the SSE-320 FVP (Ethos-U85, 256 MACs, `Sram_Only`):

| Path | Ops | Deviating > 2x their own recorded `atol` |
|---|---|---|
| Delegated to NPU | 131 | **122 (93%)** |
| Host-core fallback | 25 | **0 (0%)** |

Across all logged comparisons, 14% return exactly zero where the reference is non-zero and
65% show an error larger than half the expected magnitude. Because every host-core op is
within tolerance, the deviation is in the delegated path — the test harness, tolerance
computation, embedded reference data and runtime plumbing are all exonerated.

Four ops fail outright on this target and are recorded in `op_recipes.ETHOS_U_SKIPS` or
show as hard failures: `argmax`, `argmin` (output mismatch), `scatter`, `scatter_add`
(execute error; Vela also reports "Missing element access estimation for DMA op Scatter").

Ruled out so far:

- **Data-cache coherency.** `SystemInit` for ARMCM85 never enables the D-cache, so the
  driver's unimplemented `ethosu_flush_dcache` / `ethosu_invalidate_dcache` hooks cannot
  explain it.
- **Harness/comparison error.** Host-core fallback ops share the entire comparison path
  and are all within tolerance.

Open leads, in order:

1. **Memory-mode mismatch.** Models are exported with `memory_mode=Sram_Only` (chosen to
   avoid `invalid_weight_stream` on the all-DDR image), while `ethos_setup.cpp` sizes its
   384 KiB cache buffer for the `Shared_Sram` configuration. Try `ETHOS_MEMORY_MODE`
   /`ETHOS_SYSTEM_CONFIG` combinations and check whether the deviations collapse.
2. **FVP NPU model arguments.** `run.sh` passes
   `mps4_board.subsystem.ethosu.extra_args=--fast`; re-run without it.
3. **Per-op bisect.** Take one simple delegated op (e.g. `add`) and compare the device
   output against the host-side quantized reference element by element.

Until this is resolved, the Corstone-315 (CPU) variant is the one that gates numerical
correctness.
