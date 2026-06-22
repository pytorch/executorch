# New Op Development — QNN/HTP Backend

## Decision Tree

1. **QNN has a native op?** → Native builder approach (Steps 1–8)
2. **No native op, needs multiple QNN ops?** → Decompose pass approach

---

## Step 1: Identify the Unsupported Op

Missing ops surface as `KeyError: 'aten.my_op.default'` when running through QNN backend.

## Step 2: Check Operator Spec

- [Master Op Definitions](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/MasterOpDef.html) — IO order, params, shapes
- [HTP Op Def Supplement](https://docs.qualcomm.com/doc/80-63442-10/topic/HtpOpDefSupplement.html) — HTP-specific constraints & supported dtypes
- [Supported Ops table](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/SupportedOps.html)
- `$QNN_SDK_ROOT/include/QNN/QnnOpDef.h` — authoritative string literals
- [ATen native ops](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — PyTorch arg mapping

**⚠️ Caveats:**
- An op in the Master def may **not exist** in the HTP supplement → not available on HTP
- HTP docs may claim a dtype is supported but **fail at runtime** → always test on-device

## Step 3: Add Op Constant (`builders/qnn_constants.py`)

```python
@dataclass(init=False, frozen=True)
class OpMyOp:
    op_name: str = "MyOp"        # Must match QnnOpDef.h exactly
    param_axis: str = "axis"
    param_epsilon: str = "epsilon"
```

## Step 4: Implement Builder (`builders/op_my_op.py`)

```python
@register_node_visitor
class MyOpVisitor(NodeVisitor):
    target = ["aten.my_op.default"]  # Must be a list

    def define_node(self, node, nodes_to_wrappers):
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_wrapper = self.define_tensor(input_node, node, input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE, nodes_to_wrappers)

        output_tensor = self.get_tensor(node, node)
        output_wrapper = self.define_tensor(node, node, output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE, nodes_to_wrappers)

        op = PyQnnManager.PyQnnOpWrapper(node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpMyOp.op_name)
        op.AddInputTensors([input_wrapper])
        op.AddOutputTensors([output_wrapper])
        op.AddScalarParam(OpMyOp.param_axis, PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(axis)})
        return op  # Return None → op falls back to CPU
```

**Key patterns:**
- `QNN_TENSOR_TYPE_NATIVE` for activations, `QNN_TENSOR_TYPE_STATIC` for weights/params
- `wrapper_idx=i` for multi-output ops (tuples); companion `getitem` skip op handles indexing
- Negative dims: `dim = dim % len(shape)` (QNN requires positive axes)
- Axis remapping: `if QCOM_AXIS_ORDER in node.meta: dim = node.meta[QCOM_AXIS_ORDER].index(dim)`
- Static params: `weight = get_parameter(self.get_node(node.args[1]), self.edge_program)`
- Scalar params → `AddScalarParam`; Array params → `AddTensorParam`
- Data types: axis/dims=`UINT_32`, epsilon=`FLOAT_32`, booleans=`BOOL_8`
- Int64 index tensors: use `.to(torch.int32)` in builder + add op to `I64_IN_OPS` in `_passes/i64_to_i32.py` for CPU fallback safety (see `op_gather.py` pattern)

## Step 5: Register Builder (`builders/__init__.py`)

Add `op_my_op` to both `from . import (...)` and `__all__ = [...]` (alphabetical).

## Step 6: Add Quantizer Annotation

Add to BOTH `quantizer/annotators/htp_rules.py` AND `quantizer/annotators/lpai_rules.py`:

```python
@register_annotator([torch.ops.aten.my_op.default], QnnConstants.OpMyOp.op_name)
class MyOp(GeneralOpDef):
    pass  # Default: annotate_single_in_single_out
```

**Annotation function selection:**

| Op type | Function | When |
|---------|----------|------|
| Compute (new scale) | `annotate_single_in_single_out` | Default — most ops |
| Pass-through (`is_math_invariant`) | `annotate_in_out_obs_sharing_op` + fallback `annotate_single_in_share_out` | Reshape, Permute, Squeeze, Gather |
| Two data inputs (same quant) | Custom `annotate` with `SharedQuantizationSpec` | Scatter, where both data+src need same spec |
| Two inputs | `annotate_binary` | Add, Mul, Sub |
| Conv/Linear (weight+bias) | `annotate_conv` | Convolution, Linear |
| Skip (no QNN mapping) | `qnn_op=None` | getitem, index_copy |

**Custom multi-input annotator** (e.g., scatter where args[0] and args[3] are both data tensors):
```python
@register_annotator([torch.ops.aten.scatter.src], qnn_op=None)
class ScatterElements(GeneralOpDef):
    @staticmethod
    def annotate(node, quantization_config):
        if _is_annotated([node]): return
        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation
        if isinstance(node.args[3], Node) and _is_float_tensor(node.args[3]):
            input_qspec_map[node.args[3]] = SharedQuantizationSpec((input_act, node))
        output_qspec = SharedQuantizationSpec((input_act, node)) if _is_float_tensor(node) else None
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=output_qspec, _annotated=True)
```

## Step 7: Add Layout Transform Registration (`_passes/layout_transform.py`)

Add op to `layout_agnostic_ops` (most ops) or `layout_sensitive_ops` (conv, pool, etc.):
```python
exir_ops.edge.aten.my_op.default,
```

## Step 8: Add Unit Tests

**Model** in `tests/models.py` (alphabetical, parameterize variants):
```python
class MyOp(torch.nn.Module):
    def __init__(self, param=0):
        super().__init__()
        self.param = param
    def forward(self, x):
        return torch.my_op(x, self.param)
```

**Tests** in `tests/test_qnn_delegate.py` — add to BOTH `TestQNNFloatingPointOperator` and `TestQNNQuantizedOperator`:
```python
def test_qnn_backend_my_op(self):
    test_comb = [{
        QCOM_MODULE: [MyOp(), MyOp(param=1)],
        QCOM_SAMPLE_INPUTS: [(torch.randn(3, 4),), (torch.randn(3, 4, dtype=torch.float16),)],
    }]
    index = 0
    for comb in test_comb:
        for module in comb[QCOM_MODULE]:
            for sample_input in comb[QCOM_SAMPLE_INPUTS]:
                with self.subTest(i=index):
                    index += 1
                    self.lower_module_and_test_output(module, sample_input)
```

**Quantized test** — use separate variable to avoid overwriting module:
```python
qdq_module = self.get_qdq_module(module, sample_input)
self.lower_module_and_test_output(qdq_module, sample_input)
```

**Test data rules:**
- No duplicate indices for scatter/gather with `reduction=NONE`
- Deterministic inputs for precision-sensitive decompositions (avoid boundary values)
- Bounded inputs for ops with singularities (tan, reciprocal): `torch.rand() * 2 - 1`

**Run on-device:**
```bash
python backends/qualcomm/tests/test_qnn_delegate.py \
  -k TestQNNFloatingPointOperator.test_qnn_backend_my_op \
  --model SM8750 --host <HOST> --device <DEVICE_ID> --build_folder build-android
```

Always ask user for `--model`, `--host`, `--device`, `--build_folder` values.

## Step 9: Prevent Decomposition (if needed)

If the ATen op exists in ExecuTorch's decomp table and you have a builder for it:
- Add to `partition/utils.py` → `get_skip_decomp_table()`
- Remove from `partition/common_defs.py` → `to_be_implemented_operator` if listed there

## Step 10: Update Documentation

- `builders/README.md` — Update QNN ops table (✗ → ✓) and add to "Additional Operators Supported via Passes" table if using decomposition

---

## Decompose Pass Approach

Use when QNN has **no native op** — decompose into supported primitives.

### Approach A: Module Export
**Ref:** `_passes/decompose_linalg_vector_norm.py`. Write a `torch.nn.Module`, export, merge graph via `merge_decomposed_graph`. Simple but may produce unexpected ops.

### Approach B: Direct Graph Manipulation (RECOMMENDED)
**Ref:** `_passes/decompose_remainder.py`, `_passes/decompose_log_variants.py`.

```python
class DecomposeMyOp(ExportPass):
    def __init__(self):
        super().__init__()
        self.targets = {torch.ops.aten.my_op.default, exir_ops.edge.aten.my_op.default}

    def call(self, graph_module):
        graph = graph_module.graph
        const_cache = {}
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.targets:
                is_edge = isinstance(node.target, EdgeOpOverload)
                op = exir_ops.edge.aten.div.Tensor if is_edge else torch.ops.aten.div.Tensor
                with graph.inserting_before(node):
                    new_node = graph.create_node("call_function", op, (node.args[0],))
                    new_node.meta = copy_meta(node.meta)
                for user in node.users.copy():
                    user.replace_input_with(node, new_node)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
```

**Critical rules:** (1) handle both dialects via `EdgeOpOverload` check, (2) `copy_meta` on every new node, (3) lift scalars to tensors in edge dialect with `get_const_node`, (4) cache constants with `const_cache`, (5) for bool-output nodes use `callback=lambda m: {**m, "val": m["val"].to(torch.bool)}` in `create_node`, (6) **never pass kwargs** (like `dtype`/`device`) to `graph.create_node` for ATen ops — the ATen IR requires kwargs to be empty (`prepare_pt2e` asserts this); instead rely on `copy_meta` which propagates dtype/device via the FakeTensor in `node.meta["val"]`.

### Approach C: Built-in Decomposition Table
**Ref:** `_passes/decompose_triu.py`. Uses `make_fx` + `get_decompositions`. Only works if PyTorch has a registered decomp.

### Registration (all decompose passes)
1. `_passes/__init__.py` — import + `__all__`
2. `_passes/qnn_pass_manager.py` — The pass manager uses classmethods for pipeline definitions:
   - **Import** — add to the import block at top of file
   - **`get_annotation_passes()`** — add pass class to the returned list (runs before quantizer, ATen IR)
   - **`get_export_passes()`** — add pass class if needed for float-only path (runs after quantization, before to-edge)
   - **`get_default_pass_activations()`** — add `(PassClass, True)` ONLY if the pass also needs to run in the to-edge pipeline
   - **`get_passes_dependency_for_capture_program()`** — add `PassClass: [RemoveRedundancy]` dependency ONLY if also in `get_default_pass_activations`

**When to add to which pipeline:**
- **Annotation only** (most common for decompose passes): `get_annotation_passes()` — pass decomposes the op before the quantizer sees it
- **Export pipeline** too: if the float-only test fails without it (op doesn't get handled by PyTorch's built-in decomposition during to-edge)
- **Capture program** (to-edge) too: if the op can appear in edge dialect and needs decomposition there (e.g., `DecomposeVar`, `DecomposeCDist`, `DecomposeDiagonal`)

---

## Common Gotchas

- **Op name mismatch**: `aten.clamp`→`ReluMinMax`, `aten.expand`→`Tile`, `aten.select_copy`→`StridedSlice`. Search by functionality.
- **Multi-output ops**: Use `wrapper_idx=i` + `getitem` skip op
- **Negative dims**: QNN needs positive → `dim = dim % len(shape)`
- **QCOM_AXIS_ORDER**: `LayoutTransform` permutes NCHW→NHWC; remap axis with `.index(dim)`. `get_tensor()` auto-permutes data.
- **Int64 indices**: Add to `I64_IN_OPS` in `i64_to_i32.py` + `.to(torch.int32)` in builder (see `op_gather.py`)
- **Recompose passes**: Detect primitive sequences and replace with single native op. Ref: `recompose_pixel_unshuffle.py`
- **`partition/common_defs.py`**: Remove op from `to_be_implemented_operator` when adding support
- **HTP doc bugs**: If runtime fails but docs say supported → test on-device always.

---

## Error Debugging

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'aten.my_op.default'` | Builder not registered | Check `builders/__init__.py` + `@register_node_visitor` |
| `was not decomposed or delegated` | Op in skip decomp but partitioner rejected | Check builder `define_node` errors; check `I64_IN_OPS` |
| `QNN_GRAPH_ERROR` / `validateOpConfig failed` | HTP doesn't support config | Check params vs HTP Op Def Supplement |
| `Tensor mismatching datatypes` | Quantized: not all inputs annotated | Use custom annotator with `SharedQuantizationSpec` |
| `ValueError: Validation failed` | Wrong annotation | Check `is_math_invariant`; use `annotate_in_out_obs_sharing_op` |
| `Expected dtype int64 for index` | Op fell back to CPU with int32 index | Add to `I64_IN_OPS` + `.to(int32)` in builder |
| `Numerical mismatch` | Precision issue | Quantized: check quant params. Float: HTP FP16 precision limit |

**Debug order:** Float test first → then quantized. If float fails → builder/config issue. If only quantized fails → annotation issue.

---

## Quick Reference Checklists

**Native QNN Op:** `qnn_constants.py` → `op_my_op.py` → `builders/__init__.py` → `htp_rules.py` → `lpai_rules.py` → `layout_transform.py` → `tests/models.py` → `test_qnn_delegate.py` → `partition/utils.py` (skip decomp) → `common_defs.py` (remove to_be_implemented) → `builders/README.md`

**Decompose Pass:** `_passes/decompose_my_op.py` → `_passes/__init__.py` → `qnn_pass_manager.py` (`get_annotation_passes` + optionally `get_export_passes`; if also needed in to-edge: `get_default_pass_activations` + `get_passes_dependency_for_capture_program`) → `tests/models.py` → `test_qnn_delegate.py` → `common_defs.py` → `builders/README.md`
