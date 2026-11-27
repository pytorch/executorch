<!--
Copyright 2026 Arm Limited and/or its affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-->

# Visualize Arm models and Ethos-U performance

The Arm example scripts can open ExecuTorch PTE or TOSA graphs in
[Model Explorer](https://github.com/google-ai-edge/model-explorer). For models
run on an Ethos-U FVP, they can also overlay the measured number of cycles on
the corresponding TOSA operators.

The performance overlay helps identify expensive operators and compare the
NPU work performed by different versions of a model. It is generated from a
trace of an actual FVP execution, rather than from Vela's compile-time cycle
estimates.

## Visualize a graph

Use `--visualize_pte` to inspect the ExecuTorch program, including its delegate
calls:

```bash
./examples/arm/run.sh \
  --model_name=mv2 \
  --target=ethos-u85-256 \
  --model_explorer \
  --visualize_pte
```

Use `--visualize_tosa` to inspect the TOSA graph passed to Vela:

```bash
./examples/arm/run.sh \
  --model_name=mv2 \
  --target=ethos-u85-256 \
  --model_explorer \
  --visualize_tosa
```

These views serve different purposes. The PTE view shows the ExecuTorch
program around each delegate call. The TOSA view exposes the operators inside
the Ethos-U delegate and is the graph to which performance data can be mapped.

## Overlay measured Ethos-U cycles

Add `--perf_overlay` to a TOSA visualization:

```bash
./examples/arm/run.sh \
  --model_name=mv2 \
  --target=ethos-u85-256 \
  --model_explorer \
  --visualize_tosa \
  --perf_overlay
```

`run.sh` performs the following additional steps:

1. Enables compiler debug output so that Vela emits tables mapping command
   stream offsets back to TOSA operators.
2. Enables PMU tracing when it runs the model on the FVP.
3. Combines the trace timestamps with the Vela mapping tables.
4. Adds the resulting per-operator duration data to Model Explorer as
   `Duration (Cycles)`.

With the default build root, the relevant MobileNetV2 artifacts are:

```text
arm_test/mv2/pmu_trace.gz
arm_test/mv2/output/out_debug.xml
```

`pmu_trace.gz` contains the FVP trace events. `out_debug.xml` contains the Vela
debug tables required to attribute those events to TOSA operators. The two
files must come from the same compilation and execution.

Model Explorer color-codes operators by duration. Its node-data panel also
provides aggregate values for a selected layer, which can be used to find the
parts of the delegated graph that consume the most cycles.

## Open existing artifacts

To reopen an existing overlay without rebuilding and rerunning the model, pass
the generated files directly to `visualize.py`:

```bash
python3 examples/arm/visualize.py \
  --model_dir arm_test/mv2 \
  --tosa \
  --trace arm_test/mv2/pmu_trace.gz \
  --tables arm_test/mv2/output/out_debug.xml
```

Both `--trace` and `--tables` are required when either one is specified.

## Compare model versions

The overlay can expose performance improvements or regressions caused by
changes to model lowering, quantization, fusion, or Vela scheduling. Compare
the aggregate cycle count first. If the graph structure is unchanged, the
per-operator values can then show where the difference originated.

For a meaningful comparison, keep the following fixed between runs:

- Ethos-U target and MAC configuration
- Vela and FVP versions
- Vela system configuration and memory mode
- Input shapes and compiler options
- Quantization configuration

When a change adds, removes, or fuses operators, node identifiers may no longer
correspond between graphs. In that case, compare totals and groups of
semantically equivalent operators instead of matching nodes only by ID. Record
the PTE or source revision and verify model outputs or accuracy alongside the
cycle results.

The overlay measures work in the Ethos-U command stream. It is not an
end-to-end latency measurement and does not account for portable CPU operators
or all ExecuTorch and delegate overhead. Use
[ETDump](https://docs.pytorch.org/executorch/stable/etdump.html) when the total
runtime behavior is the metric of interest.

## Visualize VGF quantization quality

The VGF graph-wide quantization assessment can also be opened in Model
Explorer. The visualization uses the FP32 exported graph as the stable source
structure and overlays the quantized counterpart's metrics using module FQN
identity.

Install the existing Arm Model Explorer dependencies with
`examples/arm/setup.sh --enable-model-explorer`. If the setup script installed
them into the default scratch directory, make the isolated packages visible to
the Python process:

```bash
export PYTHONPATH="$PWD/examples/arm/arm-scratch/model-explorer:${PYTHONPATH:-}"
```

Then run the toy end-to-end flow:

```bash
python -m examples.arm.vgf_quantization_assessment_example \
  --model-explorer \
  --include-metadata
```

Model Explorer exposes three selectable node-data views:

- `Error (MSE)`: low error is green and the largest model-relative error is red.
- `SNR (dB)`: low SNR is red and high SNR is green.
- `Saturation (%)`: low saturation is green and high saturation is red.

Assessed FX nodes are grouped by module FQN. Selecting either a leaf node or
its FQN group shows the module identity and the corresponding precision, MSE,
SNR, cosine similarity, and saturation percentage. Parent groups retain Model
Explorer's aggregated node-data summaries, making it possible to collapse the
graph and still locate error-heavy regions.

This first visualization intentionally stops at the exported/PT2E graph. It
does not claim provenance through TOSA, Model Converter, or VGF runtime
operations; those correlations require the later stable debug/provenance work.

## Investigate quantization-sensitive modules

The Model Explorer quantization heatmap shows **where** FP32 and quantized
intermediates disagree. A low-SNR node is useful for locating numerical error,
but it does not necessarily mean that quantizing that module is responsible
for the final model-output degradation.

The VGF quantization sensitivity analyzer provides a complementary experiment.
It rebuilds the quantized model with a selected source module left in floating
point and measures whether that change improves the local tensor and the final
model output.

A small end-to-end example is available in:

```text
examples/arm/vgf_quantization_sensitivity_toy_example.py
```

Run it from the ExecuTorch repository root:

```bash
python -m examples.arm.vgf_quantization_sensitivity_toy_example
```

The example:

1. Creates a small two-layer model.
2. Exports the FP32 model.
3. Uses `VgfQuantizer` to create and calibrate an INT8 PT2E graph.
4. Runs graph-wide FP32-vs-INT8 quantization assessment.
5. Rebuilds the model with `fc1` and `fc2` independently kept in FP32.
6. Prints a JSON sensitivity report.

The example intentionally stops at the exported/PT2E graph and does not
compile or execute a VGF binary.

A typical workflow for a larger model is:

```text
Model Explorer heatmap
        |
        v
Find modules with low SNR
        |
        v
Run FP32 rescue experiment
        |
        v
Check final-output SNR / MSE improvement
```

For example, a caller can explicitly request candidate modules:

```python
report = analyze_vgf_quantization_sensitivity(
    source_model=model,
    fp32_model=fp32_export,
    baseline_quantized_model=int8_model,
    baseline_assessment=assessment,
    representative_inputs=assessment_inputs,
    build_mixed_quantized_model=build_mixed,
    candidate_module_fqns=("blocks.3.norm1",),
)
```

The mixed-model builder should rebuild and calibrate the same quantized model
while excluding the requested module from quantization, for example with:

```python
quantizer.set_module_name(module_fqn, None)
```

The report includes:

- local SNR, MSE, and cosine similarity for the candidate;
- local numerical rescue after the FP32 intervention;
- final-output SNR, MSE, and cosine changes;
- Q/DQ input-edge counts;
- whether the requested intervention was observed in the graph.

Explicit candidate module FQNs are validated against the source model before
parent expansion. Invalid names therefore fail with `ValueError` instead of
being silently ignored or interpreted as another module.

### Interpreting the result

The heatmap and sensitivity analysis answer different questions:

- **Low SNR in Model Explorer** means that a tensor differs substantially
  between FP32 and quantized execution.
- **High SNR rescue after an FP32 intervention** means that removing
  quantization from that source module improves downstream numerical quality.

A module can have poor local SNR but little downstream effect, so the
sensitivity result should be used before deciding that a layer should remain
in floating point.

The sensitivity experiment is a counterfactual mixed-precision experiment:
the model is rebuilt and recalibrated after changing the quantization policy.
It should therefore be interpreted as the effect of that mixed-precision
configuration, not as an isolated contribution of one fixed Q/DQ node.

## Current limitations

- Performance overlays are supported for the TOSA view, not the PTE view.
- The trace parser expects the gzip-compressed JSON trace generated by the
  Ethos-U FVP.
- The model directory should contain the TOSA and Vela artifacts from the same
  build as the trace.
- Cycle counts are comparable only when the target and memory timing
  configuration are equivalent.

For general information about displaying node data, see the
[Model Explorer custom node data documentation](https://github.com/google-ai-edge/model-explorer/wiki/2.-User-Guide#custom-node-data).
