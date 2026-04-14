# Observatory CLI Usage Guide

The Observatory CLI wraps any ExecuTorch export script in an Observatory context,
automatically collecting graph snapshots at each compilation stage.

## 1. Zero-Config E2E Workflow

The simplest invocation: point the CLI at your script and pass its arguments through.

```bash
python -m executorch.devtools.observatory \
    my_export_script.py [SCRIPT_ARGS...]
```

Use `--output-html` / `--output-json` to control output paths:

```bash
python -m executorch.devtools.observatory \
    --output-html /tmp/obs/report.html \
    --output-json /tmp/obs/report.json \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t
```

Use backend-specific observatory cli for additional customized lenses and hooks (qualcomm for example)

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    --output-html /tmp/obs/report.html \
    --lense_recipe=accuracy \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```

## 2. Convert JSON to HTML (Visualize Mode)

Use the `visualize` subcommand to convert an existing JSON file to HTML without
re-running the export script. This re-runs the analysis phase (lens `analyze()` methods)
against the persisted data, so HTML reports can be updated after lens code changes.

```bash
python -m executorch.backends.qualcomm.debugger.observatory visualize \
    --input-json /tmp/obs/report.json \
    --output-html /tmp/obs/report.html
```

Options:
- `--input-json` — path to the raw JSON file (required)
- `--output-html` — path for the generated HTML file (required)

## 3. Two-Step Workflow (CI collect, local visualize)

**Step 1 — CI: collect and export**
```bash
python -m executorch.backends.qualcomm.debugger.observatory \
    --output-json artifacts/report.json \
    --output-html artifacts/report.html \
    my_export_script.py --output_dir artifacts/
```

**Step 2 — Local: re-generate HTML from JSON**
```bash
python -m executorch.backends.qualcomm.debugger.observatory visualize \
    --input-json artifacts/report.json \
    --output-html artifacts/report_v2.html
```

This separates the history archive results of on-device execution (Step 1) from the interactive
visualization (Step 2), which can be re-run on demand (e.g. comparing models between 2 history commits).

## 4. Disabling Lenses via Config

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

## 5. Manual Observation Collection Points

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
## 6. Quick Reference

| Scenario | Command |
|----------|---------|
| E2E single script | `cli SCRIPT [SCRIPT_ARGS]` |
| E2E with explicit paths | `cli --output-html X.html --output-json X.json SCRIPT ...` |
| JSON -> HTML | `cli visualize --input-json X.json --output-html X.html` |
| With accuracy (backend CLI) | `backend-cli --lense_recipe=accuracy SCRIPT ...` |
