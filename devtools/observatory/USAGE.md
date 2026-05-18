# Observatory CLI Usage Guide

The Observatory CLI wraps any ExecuTorch export script in an Observatory context,
automatically collecting graph snapshots at each compilation stage.

## Modes at a glance

| Mode | Subcommand | Purpose |
|------|------------|---------|
| Collect | _(default)_ | Run a script under Observatory; export Archive (JSON) + Report (HTML). |
| Visualize | `visualize` | Re-render Report (HTML) from a previously exported Archive. |
| Compare | `compare` | Overlay multiple Archives into one Report (HTML); per-archive prefixed Region groups in the tree view. |

## 1. Zero-Config E2E Workflow

The simplest invocation: point the CLI at your script and pass its arguments through.

```bash
python -m executorch.devtools.observatory \
    my_export_script.py [SCRIPT_ARGS...]
```

Use `--output-html` / `--output-archive` to control output paths:

```bash
python -m executorch.devtools.observatory \
    --output-html /tmp/obs/report.html \
    --output-archive /tmp/obs/report.json \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t
```

> Use `--archive LABEL` to name this Archive. ``LABEL`` becomes ``Session.archive`` for every session this run produces and -- when no inner ``enter_context(region_name=...)`` is opened -- also names the default session in the dashboard sidebar.

Use a backend-specific observatory CLI for additional customised lenses and hooks (qualcomm shown):

```bash
python -m executorch.backends.xnnpack.debugger.observatory \
    --output-html /tmp/obs/report.html \
    --lens-recipe accuracy \
    examples/xnnpack/aot_compiler.py \
    --model_name=mv2 --delegate --quantize --output_dir /tmp/mv2
```

> **XNNPack note**: `examples/xnnpack/aot_compiler.py` uses relative imports (`from . import ...`).
> The CLI auto-detects this: when a `.py` path is given and its directory contains `__init__.py`,
> it runs via `runpy.run_module` instead of `runpy.run_path`. You can also pass a dotted module
> name directly (e.g. `examples.xnnpack.aot_compiler`) to force module mode explicitly.

## 2. Convert Archive to HTML (Visualize Mode)

Use the `visualize` subcommand to re-render an existing Archive (JSON) into a fresh
Report (HTML) without re-running the export script. The analysis phase
(lens `analyze()` methods) is re-run against the persisted Archive, so HTML reports
can be refreshed after lens code changes.

```bash
python -m executorch.backends.qualcomm.debugger.observatory visualize \
    --input-archive /tmp/obs/report.json \
    --output-html /tmp/obs/report.html
```

Options:
- `--input-archive` — path to the Archive (JSON) file (required).
- `--output-html` — path for the generated HTML file (required).

## 3. Compare Archives Across Backends or Runs (Compare Mode)

Use the `compare` subcommand to overlay multiple Archives into one Report (HTML).
Each Archive's records, sessions, and outermost `region_stack` entries are
prefixed with the corresponding `--label`, so identically-named pipeline stages
(e.g. "Annotated Model") stay distinct across the merged view.

```bash
python -m executorch.devtools.observatory compare \
    --input-archive xnnpack/mv2/observatory_report.json --label XNNPACK/mv2 \
    --input-archive qualcomm/mobilenet_v2/observatory_report.json --label Qualcomm/mobilenet_v2 \
    --output-html cross_backend_mv2.html \
    --title "MobileNetV2 — XNNPACK vs Qualcomm"
```

Options:
- `--input-archive` — Archive (JSON) to include (repeat once per archive).
- `--label` — display label for the **previous** `--input-archive` (must match 1:1).
- `--output-html` — destination Report (HTML) path.
- `--title` — optional page title.

In the rendered Report, toggle the **🌳 Tree** view in the left panel to see one
collapsible region per archive, then `Select` one record from each tree and click
`Compare` for a side-by-side graph diff.

## 4. Two-Step Workflow (CI collect, local visualize)

**Step 1 — CI: collect and export**
```bash
python -m executorch.backends.qualcomm.debugger.observatory \
    --output-archive artifacts/report.json \
    --output-html artifacts/report.html \
    my_export_script.py --output_dir artifacts/
```

**Step 2 — Local: re-generate HTML from the Archive**
```bash
python -m executorch.backends.qualcomm.debugger.observatory visualize \
    --input-archive artifacts/report.json \
    --output-html artifacts/report_v2.html
```

This separates the persisted Archive from the rendered Report (HTML), which can
be re-run on demand (e.g. comparing models between two history commits).

## 5. Disabling Lenses via Config

When using the Observatory Python API directly, pass a config dict to
`enter_context()` (or the legacy `enable_context()` alias) and to
`export_html_report()`:

```python
from executorch.devtools.observatory import Observatory


config = {
    "accuracy": {"enabled": False},
    "per_layer_accuracy": {"enabled": False},
}

with Observatory.enter_context("debug_run", config=config):
    # ... your export code ...

Observatory.export_html_report("report.html", config=config)
```

Config keys correspond to lens names returned by `lens.get_name()`. Each lens
checks `config.get(lens_name, {}).get("enabled", True)` during setup.

Config-only overrides (no Region label, no Session boundary) are useful for
phase-specific lens tweaks inside a Session:

```python
with Observatory.enter_context("aot"):
    Observatory.collect("a", gm)                       # region_stack=["aot"]
    with Observatory.enter_context(config={"per_layer_accuracy": {"enabled": True}}):
        # No region_name → config-only; tree view stays under "aot".
        Observatory.collect("b", gm)                    # region_stack=["aot"]
    Observatory.collect("c", gm)                        # region_stack=["aot"]
```

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
with Observatory.enter_context("pass_debug"):           # opens Session "pass_debug"
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
with Observatory.enter_context("pipeline"):
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

When running via the CLI, an Observatory context is already active.
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
| E2E single script | `cli SCRIPT [SCRIPT_ARGS]` |
| E2E with explicit paths | `cli --output-html X.html --output-archive X.json SCRIPT ...` |
| Archive → HTML | `cli visualize --input-archive X.json --output-html X.html` |
| Cross-archive overlay | `cli compare --input-archive A.json --label A --input-archive B.json --label B --output-html X.html` |
| With accuracy (backend CLI) | `backend-cli --lens-recipe accuracy SCRIPT ...` |
| With multiple recipes | `backend-cli --lens-recipe accuracy --lens-recipe adb SCRIPT ...` |
