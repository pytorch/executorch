# Observatory CLI Usage Guide

The Observatory CLI wraps any ExecuTorch export script in an Observatory context,
automatically collecting graph snapshots and accuracy metrics at each compilation stage.

## 1. Zero-Config E2E Workflow

The simplest invocation: point the CLI at your script and pass its arguments through.
Use `--report-html` to set output paths explicitly:
```bash
python -m devtools.observatory.cli \
    {your original script and arguments}
```
For example:

```bash
python -m devtools.observatory.cli \
    --report-html /tmp/obs/report.html \
    --report-json /tmp/obs/report.json \
    --report-title "Swin V2-T Qualcomm" \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t
```

Use backend-specific observatory cli for additional customized lenses and hooks (qualcomm for example)

```bash
python -m backends.xnnpack.debugger.observatory.cli \
    --report-html /tmp/obs/report.html \
    --accuracy \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```


## 2. JSON-Only Export (CI / Storage)

Use `--json-only` to collect data and export only the raw JSON, skipping HTML generation.
This is useful in CI pipelines where you want to store a compact artifact and generate
the HTML report locally later.

```bash
python -m backends.qualcomm.debugger.observatory.cli \
    --json-only \
    --report-json /tmp/obs/report.json \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t
```

The JSON file contains all raw lens digests and session data. No HTML is written.

## 3. Convert JSON to HTML (Visualize Mode)

Use the `visualize` subcommand to convert an existing JSON file to HTML without
re-running the export script. This re-runs the analysis phase (lens `analyze()` methods)
against the persisted data, so HTML reports can be updated after lens code changes.

```bash
python -m backends.qualcomm.debugger.observatory.cli visualize \
    --input /tmp/obs/report.json \
    --output /tmp/obs/report.html \
    --title "Swin V2-T Qualcomm"
```

Options:
- `--input` / `-i` — path to the raw JSON file (required)
- `--output` / `-o` — path for the generated HTML file (required)
- `--title` — report title shown in the HTML header (default: "Observatory Report")

## 4. Two-Step Workflow (CI collect, local visualize)

Combine steps 2 and 3 for a CI-collect / local-visualize pattern:

**Step 1 — CI: collect and export JSON only**
```bash
python -m backends.qualcomm.debugger.observatory.cli \
    --json-only --report-json artifacts/report.json \
    my_export_script.py --output_dir artifacts/
```

**Step 2 — Local: convert JSON to HTML**
```bash
python -m backends.qualcomm.debugger.observatory.cli visualize \
    --input artifacts/report.json \
    --output artifacts/report.html \
    --title "My Model Report"
```

This separates the history archive results of on-device execution (Step 1) from the interactive
visualization (Step 2), which can be re-run on demand (e.g. comparing models between 2 history commits).

## 5. Disabling Lenses

### Disable accuracy collection (faster runs, no accuracy metrics)

```bash
python -m backends.qualcomm.debugger.observatory.cli \
    --no-accuracy \
    my_script.py [script_args...]
```

### Skip all report output (collect only, no files written)

```bash
python -m backends.qualcomm.debugger.observatory.cli \
    --no-report \
    my_script.py [script_args...]
```

### Disable lenses via config in custom scripts

When using the Observatory Python API directly, pass a config dict to
`enable_context()` or `export_html_report()`:

```python
from executorch.devtools.observatory import Observatory


config = {
    "accuracy": {"enabled": False},
    "per_layer_accuracy": {"enabled": False},
}

with Observatory.enable_context(config=config):
    # ... your export code ...

Observatory.export_html_report("report.html", config=config)
```

Config keys correspond to lens names returned by `lens.get_name()`. Each lens
checks `config.get(lens_name, {}).get("enabled", True)` during setup.

## 6. Manual Observation Collection Points

You can insert `Observatory.collect()` calls anywhere in your code to capture
intermediate graph states. This is useful for debugging pass transforms or
custom lowering steps.

### Basic usage

```python
import torch
from executorch.devtools.observatory import Observatory

model = MyModel().eval()
graph = torch.fx.symbolic_trace(model)

Observatory.clear()
with Observatory.enable_context():
    Observatory.collect("original", graph)

    # Apply a pass
    transformed = my_pass(graph)
    Observatory.collect("after_my_pass", transformed)

Observatory.export_html_report("pass_debug.html")
Observatory.export_json("pass_debug.json")
```

### Pass transform debugging

Use `observe_pass` to automatically collect graphs before and after a pass.
Wrap any `PassBase` subclass instance, callable, or use it as a class decorator:

```python
from executorch.devtools.observatory import Observatory, observe_pass
from executorch.exir.passes.remove_graph_asserts_pass import RemoveGraphAssertsPass

# Wrap pass instances — default collects both input and output graphs
pass_a = observe_pass(RemoveGraphAssertsPass())
pass_b = observe_pass(MyCustomPass())

Observatory.clear()
with Observatory.enable_context():
    result_a = pass_a(graph_module)
    # collects "RemoveGraphAssertsPass/input" and "RemoveGraphAssertsPass/output"

    result_b = pass_b(result_a.graph_module)
    # collects "MyCustomPass/input" and "MyCustomPass/output"

    # Call again — names auto-deduplicate
    result_c = pass_a(result_b.graph_module)
    # collects "RemoveGraphAssertsPass/input #2" and "RemoveGraphAssertsPass/output #2"

Observatory.export_html_report("pass_debug.html")
```

Control what is collected with boolean flags:

```python
# Collect only the output graph
observed = observe_pass(SomePass(), collect_input=False)

# Collect only the input graph
observed = observe_pass(SomePass(), collect_output=False)

# Override the record name
observed = observe_pass(SomePass(), name="step_1")
```

Use as a class decorator to make all instances observable:

```python
@observe_pass
class MyPass(PassBase):
    def call(self, gm):
        # ... transform logic ...
        return PassResult(gm, True)

# Or with parameters:
@observe_pass(name="Quantize", collect_input=False)
class QuantizePass(PassBase):
    def call(self, gm):
        ...
```

`observe_pass` is a no-op when no Observatory context is active.

### Inside the CLI-wrapped script (zero-code-change)

When running via the CLI, `Observatory.enable_context()` is already active.
You can add collection points to your script without any setup:

```python
# In your export script (e.g., my_model.py):
from executorch.devtools.observatory import Observatory

# This fires only when Observatory context is active (i.e., when run via CLI).
# It is a no-op otherwise.
Observatory.collect("pre_quantize", exported_program)
```
## 7. Quick Reference

| Scenario | Command |
|----------|---------|
| E2E single script | `cli script.py [script_args]` |
| E2E with explicit paths | `cli --report-html X.html --report-json X.json script.py ...` |
| JSON only (no HTML) | `cli --json-only --report-json X.json script.py ...` |
| JSON → HTML | `cli visualize --input X.json --output X.html` |
| No accuracy metrics | `cli --no-accuracy script.py ...` |
| No output files | `cli --no-report script.py ...` |
