# Observatory (GraphView Minimal)

This directory provides a new, review-focused Observatory implementation that follows the GraphView RFC contracts while keeping behavior close to the existing `debugging_utils` UI.

## Goals
1. Keep implementation simple and easy to review.
2. Preserve original observatory report behavior where practical.
3. Make graph rendering first-class via `GraphView` and `GraphHub`.
4. Keep JS runtime code split into topic files under `templates/js`.
5. Support ETRecord auto-collection through monkey patching while context is enabled.

## Main Files
1. `interfaces.py`: RFC-style contracts (`ViewBlock`, `ViewList`, `GraphView`, `Lens`, `Frontend`).
2. `observatory.py`: context lifecycle, collection, analysis, report payload assembly.
3. `graph_hub.py`: graph asset/layer registry for `graph_ref`-based reuse.
4. `auto_collect.py`: ETRecord monkey-patch install/uninstall.
5. `lenses/graph.py`: canonical base graph producer (`GraphLens`).
6. `lenses/metadata.py` and `lenses/stack_trace.py`: minimal migrated lenses.
7. `templates/js/*.js`: split UI runtime logic.
8. `templates/css/main.css`: UI styling baseline.

## Quick Start

```bash
source ~/executorch/.venv/bin/activate
source ~/executorch/qairt/2.37.0.250724/bin/envsetup.sh
export PYTHONPATH=~/
```

```python
from executorch.backends.qualcomm.debugger.observatory import Observatory

with Observatory.enable_context():
    # collect your graph artifacts
    ...

Observatory.export_html_report("/tmp/observatory_report.html")
```

## Block Contract Example

```python
from executorch.backends.qualcomm.debugger.observatory.interfaces import ViewBlock, ViewList

return ViewList(
    blocks=[
        ViewBlock(
            id="summary",
            title="Summary",
            type="table",
            record={"data": {"a": 1}},
            compare={"mode": "auto"},
        )
    ]
)
```

## GraphView Example

```python
from executorch.backends.qualcomm.debugger.observatory.interfaces import GraphView

view = GraphView(
    id="acc_graph",
    title="Accuracy Graph",
    graph_ref="record_0",
    default_layers=["accuracy/error"],
    default_color_by="accuracy/error",
)
block = view.as_block()
```

## ETRecord Auto-Collection

`Observatory.enable_context()` installs temporary monkey patches on ETRecord methods:
1. `ETRecord.add_exported_program`
2. `ETRecord.add_edge_dialect_program`
3. `ETRecord.add_extra_export_modules`

These calls automatically trigger Observatory collection while context is active. Patches are removed on outermost context exit.

## Demos
1. `examples/demo_graphview_accuracy_compare.py`
   - Graph compare with per-layer accuracy overlay.
   - Supports `--model toy` and `--model swin`.
2. `examples/demo_etrecord_auto_collect.py`
   - Demonstrates zero manual `collect()` for ETRecord paths.
3. `examples/generate_ui_test_harness.py`
   - Generates an interactive HTML harness for JS/UI test cases.

## JS Runtime Layout
1. `templates/js/00_state.js`: state bootstrap.
2. `templates/js/01_utils.js`: utilities + graph payload helpers.
3. `templates/js/02_layout.js`: shell and index rendering.
4. `templates/js/03_blocks.js`: block and compare rendering.
5. `templates/js/04_actions.js`: navigation/theme/selection actions.
6. `templates/js/05_bootstrap_api.js`: init + `window.ObservatoryAPI` + delegated actions.

## Test Plan
1. Unit-level Python smoke:
   - collect toy graph and export HTML.
2. ETRecord injection smoke:
   - run `demo_etrecord_auto_collect.py` and verify records are captured.
3. GraphView compare smoke:
   - run `demo_graphview_accuracy_compare.py` and compare records in report UI.
4. Interactive JS harness:
   - run `generate_ui_test_harness.py` and verify block rendering and actions.
