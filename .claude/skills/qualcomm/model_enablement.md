# Model Enablement

Checklist for enabling a new model end-to-end on the QNN backend.

---

## 1. Identify Unsupported Ops

Export the model and check which ops fall back to CPU:

```python
from executorch.backends.qualcomm.utils.utils import capture_program

prog = capture_program(model, example_inputs)
for node in prog.exported_program.graph.nodes:
    if node.op == "call_function":
        print(node.target.__name__)
```

Or run the full lowering and inspect the partition result — nodes outside the delegate are CPU fallbacks.

For each unsupported op, follow `new_op_development.md`.

---

## 2. Add Export Script

Place the script under `examples/qualcomm/scripts/<model_name>.py`. Use `build_executorch_binary` as the standard entry point:

```python
from executorch.examples.qualcomm.utils import build_executorch_binary

build_executorch_binary(
    model=model,
    inputs=example_inputs,
    soc_model=args.model,
    file_name=f"{args.artifact}/{pte_filename}",
    dataset=calibration_data,       # None for FP16
    quant_dtype=QuantDtype.use_8a8w, # omit for FP16
    online_prepare=args.online_prepare,
)
```

For models requiring custom runners, add under `examples/qualcomm/oss_scripts/`.

---

## 3. Verify Delegation

After lowering, confirm the graph is fully delegated:

```python
from executorch.backends.qualcomm.utils.utils import draw_graph

draw_graph("model_graph", prog.exported_program.graph)
```

Expected: all compute nodes inside a single `torch.ops.higher_order.executorch_call_delegate` node. Any remaining `call_function` nodes are CPU fallbacks — investigate and fix.

---

## 4. Add Model-Level Tests

In `tests/test_qnn_delegate.py`, add to `TestQNNFloatingPointModel` and/or `TestQNNQuantizedModel`:

```python
def test_qnn_backend_my_model(self):
    # setup model and inputs
    module = MyModel()
    sample_input = (torch.randn(1, 3, 224, 224),)
    # lower and test
    self.lower_module_and_test_output(module, sample_input)
```

For script-based tests (with artifact dependencies), add to `TestExampleScript` or `TestExampleOssScript`.

---

## 5. Accuracy Validation

Run on device and compare outputs against CPU reference:

```python
import torch

cpu_output = model(*example_inputs)
qnn_output = # load from device execution

torch.testing.assert_close(qnn_output, cpu_output, rtol=1e-2, atol=1e-2)
```

Typical tolerances:
- FP16: `rtol=1e-2, atol=1e-2`
- INT8 quantized: `rtol=1e-1, atol=1e-1` (accuracy depends on calibration quality)

---

## 6. Common Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| Op falls back to CPU | Missing builder or annotation | Add op builder + quantizer annotation |
| Shape mismatch after layout transform | NHWC/NCHW confusion | Check `LayoutTransform` pass, verify `get_tensor` axis order |
| Quantization accuracy degraded | Poor calibration data | Use representative dataset; try per-channel quantization |
| `KeyError` in `node_visitors` | Builder not registered | Check `builders/__init__.py` import |
| Context binary compile failure | QNN op spec mismatch | Verify IO order and parameter names against `QnnOpDef.h` |
| `online_prepare` vs offline mismatch | Context binary format | Use `--online_prepare` for QAIRT Visualizer; offline for deployment |
