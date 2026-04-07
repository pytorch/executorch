# Observatory CLI Usage Guide

The Observatory CLI wraps any ExecuTorch export script in an Observatory context,
automatically collecting graph snapshots and accuracy metrics at each compilation stage.

## 1. Zero-Config E2E Workflow

The simplest invocation: point the CLI at your script and pass its arguments through.
The CLI infers the output directory from the script's `-a`/`--artifact` or `-o`/`--output_dir` flag.

```bash
python -m backends.qualcomm.debugger.observatory.cli \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t
```

Output (inferred from `-a ./swin_v2_t`):
- `./swin_v2_t/observatory_report.html` — interactive report
- `./swin_v2_t/observatory_report.json` — raw data for later re-analysis

To set paths explicitly:

```bash
python -m backends.qualcomm.debugger.observatory.cli \
    --report-html /tmp/obs/report.html \
    --report-json /tmp/obs/report.json \
    --report-title "Swin V2-T Qualcomm" \
    examples/qualcomm/oss_scripts/swin_v2_t.py \
    --model SM8650 -b ./build-android -d imagenet-mini/val -a ./swin_v2_t
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

This separates the expensive on-device execution (Step 1) from the interactive
visualization (Step 2), which can be re-run any number of times.

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
from executorch.backends.qualcomm.debugger.observatory import Observatory

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
from executorch.backends.qualcomm.debugger.observatory import Observatory

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

To compare graphs before and after a specific pass:

```python
with Observatory.enable_context():
    for name, module in model.named_modules():
        before = torch.fx.symbolic_trace(module)
        Observatory.collect(f"{name}/before", before)

        after = my_transform(before)
        Observatory.collect(f"{name}/after", after)
```

### Inside the CLI-wrapped script (zero-code-change)

When running via the CLI, `Observatory.enable_context()` is already active.
You can add collection points to your script without any setup:

```python
# In your export script (e.g., my_model.py):
from executorch.backends.qualcomm.debugger.observatory import Observatory

# This fires only when Observatory context is active (i.e., when run via CLI).
# It is a no-op otherwise.
Observatory.collect("pre_quantize", exported_program)
```

## 7. Demo Script Modes

The batch demo script (`scripts/generate_observatory_demo.py`) supports three modes:

### Default: run all jobs

```bash
python scripts/generate_observatory_demo.py \
    --xnn-models mv2 \
    --qualcomm-models mobilenet_v2 \
    --qnn-sdk-root /path/to/qairt/2.37.0
```

Runs each job, writes HTML + JSON reports, and refreshes `index.html`.

### `--plan-only`: register without running

```bash
python scripts/generate_observatory_demo.py \
    --plan-only \
    --xnn-models mv2,resnet18 \
    --qualcomm-models mobilenet_v2,roberta
```

Creates output directories and writes `manifest.json` + `index.html` with all
jobs listed as `"planned"` status. No scripts are executed. Useful for previewing
the job plan or pre-creating the index before a long run.

### `--visualize-only`: re-render HTML from existing JSON

```bash
python scripts/generate_observatory_demo.py --visualize-only
```

Reads `manifest.json`, calls `cli visualize` for each job that has an existing
JSON file, and refreshes `index.html`. Jobs without a JSON file are skipped with
a warning. Requires a prior successful run (or manually placed JSON files).

Use this after updating lens code to regenerate all HTML reports without
re-running the expensive export scripts.

## 8. Quick Reference

| Scenario | Command |
|----------|---------|
| E2E single script | `cli script.py [script_args]` |
| E2E with explicit paths | `cli --report-html X.html --report-json X.json script.py ...` |
| JSON only (no HTML) | `cli --json-only --report-json X.json script.py ...` |
| JSON → HTML | `cli visualize --input X.json --output X.html` |
| No accuracy metrics | `cli --no-accuracy script.py ...` |
| No output files | `cli --no-report script.py ...` |
| Batch plan (no run) | `generate_observatory_demo.py --plan-only` |
| Batch run | `generate_observatory_demo.py` |
| Batch re-render HTML | `generate_observatory_demo.py --visualize-only` |
| Single model batch | `generate_observatory_demo.py --xnn-models mv2 --qualcomm-models mobilenet_v2` |
