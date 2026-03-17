# Observatory GraphView Minimal Redesign Plan (Context Checkpoint)

## Branch and Scope
- Working branch: `dev1/boyuc/observatory_graphview_minimal`
- Base context: includes prior `fx_viewer` API refinement commits.
- Final rebase target requested by user: `MLG/dev`.
- Non-goal for this branch: delete old `backends/qualcomm/debugger/debugging_utils`; keep old infra intact and introduce a new review-focused implementation under `backends/qualcomm/debugger/observatory`.

## User-Approved Reuse Patterns
1. Reuse existing monkey-patch lifecycle pattern for auto collection.
2. Reuse `fx_viewer` runtime APIs for graph mount/compare.
3. Keep RFC class structure (explicitly requested) instead of custom simplified dispatch class hierarchy.
4. Keep serialization/safe-call helpers centralized.

## Design Constraints From User
1. Preserve original observatory JS/HTML behavior as much as possible.
2. Split JS into separate template files for readability/review.
3. Use RFC contracts (`ViewBlock`, `ViewList`, `GraphView`, `GraphHub`, `GraphLens`) in the new infra.
4. Migrate only minimal lenses at first: metadata and stack trace.
5. No manual `observe()` insertion for ETRecord collection; use monkey-patching to auto insert collection points.
6. Add demo + tutorial/test plan, including swIN-style flow and compare support.

## Source References Reviewed
- RFC:
  - `backends/qualcomm/debugger/debugging_utils/RFC_OBSERVATORY_GRAPHVIEW_INTEGRATION.md`
  - `backends/qualcomm/debugger/debugging_utils/FX_VIEWER_ACCURACY_INTEGRATION_PLAN.md`
- Existing observatory infra:
  - `backends/qualcomm/debugger/debugging_utils/observatory.py`
  - `backends/qualcomm/debugger/debugging_utils/interfaces.py`
  - `backends/qualcomm/debugger/debugging_utils/html_template.py`
  - `backends/qualcomm/debugger/debugging_utils/extensions/metadata.py`
  - `backends/qualcomm/debugger/debugging_utils/extensions/stack_trace.py`
  - `backends/qualcomm/debugger/debugging_utils/extensions/adb_execution.py`
- ETRecord integration points:
  - `devtools/etrecord/_etrecord.py`
- fx_viewer embedding/runtime:
  - `backends/qualcomm/utils/fx_viewer/README.md`
  - `backends/qualcomm/utils/fx_viewer/exporter.py`
  - `backends/qualcomm/utils/fx_viewer/templates/*.js`
- Existing examples for style and flow:
  - `examples/qualcomm/oss_scripts/swin_transformer.py`
  - `backends/qualcomm/utils/fx_viewer/examples/demo_per_layer_accuracy_fx.py`

## Planned Commit Series
1. `observatory(core): scaffold RFC contracts and package layout`
   - Add `observatory/interfaces.py`, `graph_hub.py`, package init files.
2. `observatory(lenses): add minimal metadata/stack_trace lenses`
   - New lenses under `observatory/lenses/`.
3. `observatory(graph): add GraphLens and graph asset assembly`
   - Graph extraction from `GraphModule`/`ExportedProgram` using `FXGraphExporter`.
4. `observatory(ui): split template JS into topic files and preserve current behavior`
   - Move old single JS string logic to `observatory/templates/js/*.js`.
   - Keep feature parity with old index/dashboard/compare flow.
5. `observatory(auto-collect): ETRecord monkey patch auto collection`
   - Install/uninstall patch from context lifecycle.
6. `observatory(demo): add swin-style and GraphView compare demos`
   - Add scripts in `observatory/examples/`.
7. `observatory(tests-docs): add tutorial-like test plan and usage docs`
   - Add focused test cases and README notes.

## Execution Checklist
- [ ] Build new observatory runtime and report export path.
- [ ] Verify record view and compare view behavior parity against old template baseline.
- [ ] Verify `GraphView` blocks mount `fx_viewer` correctly.
- [ ] Verify compare-mode graph sync toggle behavior.
- [ ] Verify ETRecord monkey-patch auto collects when context enabled.
- [ ] Run demo scripts with env setup:
  - `source ~/executorch/.venv/bin/activate`
  - `source ~/executorch/qairt/2.37.0.250724/bin/envsetup.sh`
- [ ] Capture commands/outcomes in docs.

## Risk Controls
1. Keep old module untouched for rollback and behavior comparison.
2. Stage changes as small commits with runnable checkpoints.
3. Minimize logic churn in first UI pass; mostly mechanical split + explicit API wrappers.
4. Keep unresolved ambiguities in `Questions.md` and proceed with conservative defaults.
