---
name: cortex-m
description: Build, test, or develop the Cortex-M (CMSIS-NN) backend. Use when working on backends/cortex_m/, running Cortex-M tests, or exporting models for Cortex-M targets.
---

# Cortex-M (CMSIS-NN) Backend

## Architecture

Not a delegate backend — no partitioner. Custom ops and graph passes replace ATen quantized ops with CMSIS-NN equivalents at the graph level.

## Pipeline

Uses standard PT2E quantization (`prepare_pt2e` / `convert_pt2e`), then `CortexMPassManager` rewrites quantized ops to `cortex_m::` equivalents.

```python
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

quantizer = CortexMQuantizer()
captured = export(model, example_inputs).module()
prepared = prepare_pt2e(captured, quantizer)
prepared(*example_inputs)  # calibration
quantized = convert_pt2e(prepared)

exported = export(quantized, example_inputs)
edge = to_edge_transform_and_lower(
    exported,
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
edge._edge_programs["forward"] = CortexMPassManager(
    edge.exported_program(), CortexMPassManager.pass_list
).transform()
et_program = edge.to_executorch()
```

In tests, `CortexMTester` wraps this pipeline:

```python
from executorch.backends.cortex_m.test.tester import CortexMTester

tester = CortexMTester(model, example_inputs)
tester.quantize().export().to_edge().run_passes().to_executorch()
```

## Key Files

| File | Purpose |
|------|---------|
| `backends/cortex_m/quantizer/quantizer.py` | `CortexMQuantizer` — quantizes model for CMSIS-NN |
| `backends/cortex_m/passes/cortex_m_pass_manager.py` | `CortexMPassManager` — rewrites ATen ops → `cortex_m::` ops |
| `backends/cortex_m/test/tester.py` | `CortexMTester` — test harness with `test_dialect()` and `test_implementation()` |
| `backends/cortex_m/ops/operators.py` | Python op definitions and reference implementations (`cortex_m::` namespace) |
| `backends/cortex_m/ops/operators.yaml` | C++ kernel registration schemas (used by build system) |

C++ kernels calling CMSIS-NN APIs live under `backends/cortex_m/ops/`.

## Testing

**Toolchain setup (required for `test_implementation` tests):**
```bash
./examples/arm/setup.sh --i-agree-to-the-contained-eula
source ./examples/arm/arm-scratch/setup_path.sh
```

**Run all tests:**
```bash
source ./examples/arm/arm-scratch/setup_path.sh
pytest backends/cortex_m/test/
```

`test_dialect_*` tests verify graph correctness (pure Python, no toolchain needed).
`test_implementation_*` tests verify numerical accuracy on the Corstone-300 FVP (requires toolchain on PATH).

**Baremetal build:**
```bash
backends/cortex_m/test/build_test_runner.sh
```

## Adding a New Op

1. Define the op schema, meta function, and reference implementation in `operators.py`
2. Write the C++ kernel in `backends/cortex_m/ops/` calling CMSIS-NN APIs
3. Register the `.out` kernel in `operators.yaml`
4. Add a pass to rewrite the ATen op → `cortex_m::` op
5. Test with `CortexMTester.test_dialect()` (graph correctness) and `test_implementation()` (numerical accuracy on FVP)
