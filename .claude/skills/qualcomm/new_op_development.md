# New Op Development

Full reference: `backends/qualcomm/builders/README.md` (op builder) and `backends/qualcomm/quantizer/README.md` (quantizer annotation).

## Overview

Adding a new op requires three steps:
1. Implement the op builder (`builders/op_*.py`)
2. Register quantizer annotation (`quantizer/annotators/`)
3. Add unit tests (`tests/`)

**Important**: If the torch op requires **multiple QNN ops** to implement (e.g., no direct QNN equivalent), use a **decompose pass** instead of creating multiple ops in a single builder. Skip Steps 3–6 and follow the **Decompose Pass Approach** section at the bottom of this file.

---

## Step 1: Identify the Unsupported Op

Run the model through the QNN backend. A missing op surfaces as:

```
KeyError: 'aten.native_layer_norm.default'
```

To trace back to the source PyTorch layer:

```python
from executorch.backends.qualcomm.utils.utils import capture_program

prog = capture_program(MyModel(), example_inputs)
for node in prog.exported_program.graph.nodes:
    if node.op == "call_function" and node.target.__name__ == 'aten.native_layer_norm.default':
        print(node.meta["source_fn_stack"])
```

---

## Step 2: Check Operator Spec

- **QNN side**: [Operator Definitions](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/MasterOpDef.html) — check IO order, required vs optional tensors, parameter names and shapes
- **PyTorch side**: [ATen Operator Definitions](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — map PyTorch args to QNN IO/params
- **Fallback search**: [Supported Ops table](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/SupportedOps.html)
- **Header reference**: `$QNN_SDK_ROOT/include/QNN/QnnOpDef.h` — authoritative string literals

---

## Step 3: Add Op Constant

In `builders/qnn_constants.py`, add a dataclass (alphabetical order):

```python
@dataclass(init=False, frozen=True)
class OpLayerNorm:
    op_name: str = "LayerNorm"
    param_epsilon = "epsilon"
    param_axes = "axes"
```

String values must exactly match `QnnOpDef.h`.

---

## Step 4: Implement the Builder

Create `builders/op_layer_norm.py`:

```python
import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA
from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpLayerNorm, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter

@register_node_visitor
class LayerNormVisitor(NodeVisitor):
    target = ["aten.native_layer_norm.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(self, node, nodes_to_wrappers):
        # 1. Input activation
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node, node, input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # 2. Weight (gamma) and bias (beta) — STATIC tensors
        weight_node = self.get_node(node.args[2])
        weight_tensor = get_parameter(weight_node, self.edge_program)
        weight_tensor_wrapper = self.define_tensor(
            weight_node, node, weight_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        bias_node = self.get_node(node.args[3])
        bias_tensor = get_parameter(bias_node, self.edge_program)
        bias_tensor_wrapper = self.define_tensor(
            bias_node, node, bias_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        # 3. Parameters
        normalized_shapes = node.args[1]
        if len(normalized_shapes) != 1:
            print("QNN only supports normalized output with rank 1")
            return
        axes = [len(input_tensor.shape) - 1]
        axes_shape = [len(axes)]
        epsilon = node.args[4]

        # 4. Output
        output_tensor = self.get_tensor(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node, node, output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # 5. Build op
        layer_norm_op = PyQnnManager.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpLayerNorm.op_name,
        )
        layer_norm_op.AddInputTensors(
            [input_tensor_wrapper, weight_tensor_wrapper, bias_tensor_wrapper]
        )
        layer_norm_op.AddOutputTensors([output_tensor_wrapper])
        layer_norm_op.AddScalarParam(
            OpLayerNorm.param_epsilon,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(epsilon)},
        )
        layer_norm_op.AddTensorParam(
            OpLayerNorm.param_axes,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(axes_shape), axes_shape,
            np.array(axes, dtype=np.uint32),
            True,
        )
        return layer_norm_op
```

Key notes:
- `target` must be a list (multiple targets can share one visitor)
- Use `QNN_TENSOR_TYPE_NATIVE` for activations, `QNN_TENSOR_TYPE_STATIC` for weights/biases
- `define_tensor` handles `APP_READ`/`APP_WRITE` detection internally — always pass `NATIVE`
- `wrapper_idx` needed when node output is a tuple (e.g. split ops)
- Return `None` to signal validation failure → op falls back to CPU

---

## Step 5: Register the Builder

In `builders/__init__.py` (alphabetical order):

```python
from . import (
    ...
    op_layer_norm,
    ...
)
__all__ = [..., op_layer_norm, ...]
```

---

## Step 6: Add Quantizer Annotation

In `quantizer/annotators/{backend}_rules.py`:

```python
@register_annotator(
    [torch.ops.aten.native_layer_norm.default],
    QnnConstants.OpLayerNorm.op_name,
)
class LayerNormAnnotator(GeneralOpDef):
    @staticmethod
    def annotate(node, quantization_config):
        annotate_single_in_single_out(node, quantization_config)
```

- Use `qnn_op=None` for skip ops (e.g. `operator.getitem`)
- `annotate_single_in_single_out` covers most cases; implement custom logic for multi-input ops

Full annotation tutorial: `backends/qualcomm/quantizer/README.md`

### Choosing the right annotate function

The QNN backend validates quantization constraints via `backend_opinfo` (QNN SDK ≥ 2.41). If validation fails with:

```
ValueError: Validation failed for node <name> with target aten.<op>.default
```

Check the warning log above it — it will say which constraint failed. The most common case is `is_math_invariant=True`, which means the op does not change values (only rearranges data), so input and output **must share the same quantization parameters**.

| Op type | annotate function | Example ops |
|---------|-------------------|-------------|
| General (input → output with new scale) | `annotate_single_in_single_out` | LayerNorm, Conv2d |
| Pass-through (rearranges data only) | `annotate_in_out_obs_sharing_op` + fallback | Reshape, ChannelShuffle, PixelShuffle |
| Multi-input | `annotate_binary` | Add, Mul |

For **pass-through ops** (reshape, shuffle, permute — ops where `is_math_invariant=True`), override `annotate` like this:

```python
@register_annotator(
    [torch.ops.aten.channel_shuffle.default], QnnConstants.OpChannelShuffle.op_name
)
class ChannelShuffle(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_share_out(node, quantization_config)
```

`annotate_in_out_obs_sharing_op` shares the input's observer with the output (satisfies `is_math_invariant`). The fallback `annotate_single_in_share_out` handles the case where the input node is not yet annotated.

---

## Step 7: Add Unit Tests

In `tests/models.py` (alphabetical order):

```python
class LayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm([768], eps=1e-6)

    def forward(self, x):
        return self.layer_norm(x)
```

In `tests/test_qnn_delegate.py`, add to both `TestQNNFloatingPointOperator` and `TestQNNQuantizedOperator` (alphabetical order):

```python
def test_qnn_backend_layer_norm(self):
    module = LayerNorm()
    sample_input = (torch.randn(196, 768),)
    module = self.get_qdq_module(module, sample_input)  # quantized only
    self.lower_module_and_test_output(module, sample_input)
```

Expected result: 1 delegated node, only placeholders/output nodes remain outside the delegate.

---

## Step 8: Prevent Decomposition (if needed)

Some torch ops are in ExecuTorch's default decomposition table and will be broken into primitives **before** the QNN partitioner sees them. If QNN has a native op for it, you must explicitly skip decomposition.

**Check first** with a quick Python snippet (run from the executorch root with the `executorch` conda env active):

```python
import torch
from executorch.exir.tracer import _default_decomposition_table

decomp_table = _default_decomposition_table()
op = torch.ops.aten.channel_shuffle.default
print(op in decomp_table)  # True → will be decomposed
```

Output:
```
True  # in ExecuTorch decomp table
```

If `True`, add the op to `get_skip_decomp_table()` in `partition/utils.py` (alphabetical order):

```python
def get_skip_decomp_table() -> List[torch._ops.OperatorBase]:
    do_not_decompose = [
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.channel_shuffle.default,   # ← add here
        torch.ops.aten.col2im.default,
        ...
    ]
```

**Verification**: After adding, re-run the tests. The partitioner log should show:

```
[QNN Partitioner Op Support]: aten.channel_shuffle.default | True
```

If the op was decomposed (not in skip table), the partitioner would never see `aten.channel_shuffle.default` and the test would still pass but via decomposed primitives — not the native QNN op.

---

## Decompose Pass Approach (for ops without direct QNN equivalent)

When a torch op has **no direct QNN equivalent** and requires multiple QNN ops to implement, use a **decompose pass** to rewrite the graph into primitive ops that QNN already supports. This is preferred over creating multiple ops in a single builder.

**Reference**: `backends/qualcomm/_passes/decompose_linalg_vector_norm.py`

### Pattern

```python
# 1. Define a torch.nn.Module that implements the op using supported primitives
class MyOpDecomposed(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def forward(self, x):
        # Use only ops that QNN supports
        return torch.some_supported_op(x, self.param)


# 2. Create the ExportPass
class DecomposeMyOp(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.target == torch.ops.aten.my_op.default:
                param = node.args[1]  # extract params from node
                model = MyOpDecomposed(param)
                ep = torch.export.export(model, (node.args[0].meta["val"],), strict=True)
                decomposed_module = ep.run_decompositions().graph_module

                with graph.inserting_before(node):
                    remap = {"x": node.args[0]}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                    )
                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
```

### Registration

1. Add to `_passes/__init__.py` (alphabetical order):
   ```python
   from .decompose_my_op import DecomposeMyOp
   ```

2. Add to `_passes/qnn_pass_manager.py` imports and both pipeline methods:
   - `transform_for_annotation_pipeline` (before quantizer)
   - `transform_for_export_pipeline` (before `to_edge`)

3. Remove the op from `to_be_implemented_operator` in `partition/common_defs.py`

### Notes
- The decomposed module must only use ops that QNN already supports
- `ep.run_decompositions()` ensures the graph is in edge IR form
- `remap` maps placeholder names in the decomposed graph to actual nodes in the target graph
- No separate quantizer annotation needed — the decomposed ops already have their own annotations
