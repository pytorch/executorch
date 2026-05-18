# RFC: Sessions as a First-Class Unit (and the Archive grouping above them)

| Field | Value |
| --- | --- |
| Status | Initial commit; replaces the singleton "Run Dashboard" worldview. |
| Authors | observatory authors |
| Affected versions | first release that ships this contract |
| Worldview vocabulary | Archive ⊃ Session ⊃ Region ⊃ Record (cf. README §worldview, RFC §4.4) |

## 1. Motivation

Until this RFC the report had a *singleton* dashboard concept: regardless of how many sessions a run produced, or how many archives a comparison report aggregated, the HTML had exactly one "📊 Run Dashboard" panel. The framework reflected this with a flat `SessionResult.start_data` / `end_data` view that merged every session's per-lens payload into one dict (last-write-wins). That worked when:

* a CLI run emitted exactly one outermost `enter_context`, **and**
* the report was assembled from a single archive.

It fell over the moment either assumption broke:

| Scenario | Symptom |
| --- | --- |
| Compare report (≥ 2 archives) | The flat `start_data` / `end_data` were keyed by `<label>/<lens>` to avoid collisions, but every dashboard frontend looked them up by bare lens name (`session.start_data["adb"]`). Result: empty ADB / accuracy panels even though the data was on disk. |
| Multi-session collection | Even single-archive, the second session's payload silently overwrote the first in the flat dict. |
| Compare-mode region tree | `compare_archives` prepended the archive label to every record's `region_stack`, so the tree double-encoded the archive (label-as-region + label-prefixed record names + label-prefixed session ids). |

This RFC removes the singleton worldview. **Sessions become a first-class addressable unit; Archive is a separate grouping above sessions only used in compare mode.**

## 2. Worldview

```
Archive
  ├── Session "qualcomm/swin_v2"
  │     ├── Region "quantization" → Records ...
  │     ├── Region "edge" → Records ...
  │     └── Region "device" → Records ...
  └── Session ... (rare; one outermost per enter_context)
```

* **Archive** — a unit of persistence. Each `--output-archive` run writes one archive. Single-archive reports have exactly one archive labelled `"default"`; compare reports have N archives (one per `--input-archive`).
* **Session** — the heavyweight scope opened by an outermost `enter_context()`. Carries `id`, `name`, `archive`, `start_ts`, `end_ts`, plus per-lens `start_data` / `end_data` payloads from `on_session_start` / `on_session_end`. Most CLI runs produce exactly one Session.
* **Region** — lightweight named scope; nests freely; pure labelling; no lens hooks fire at Region boundaries.
* **Record** — one `Observatory.collect()` item; carries `session_id`, `region_stack` (snapshot at collect time), and a per-lens digest map.

## 3. Payload contract (HTML report)

```jsonc
{
  "title": "...",
  "generated_at": "...",

  "archives": [                                  // NEW: top-level grouping
    { "label": "XNNPACK/mv2",
      "session_ids": ["XNNPACK/mv2/default"] },
    { "label": "Qualcomm/mobilenet_v2",
      "session_ids": ["Qualcomm/mobilenet_v2/default"] }
  ],

  "sessions": [                                  // NEW: flat list, indexed by id
    { "id":   "XNNPACK/mv2/default",
      "name": "default",
      "archive": "XNNPACK/mv2",                  // <- ties session into archive
      "start_ts": 1731.123, "end_ts": 1734.456,
      "start_data": { "metadata": {...}, "adb": {...} },
      "end_data":   { "metadata": {...}, "adb": {...} } },
    ...
  ],

  "records": [
    { "name": "Annotated Model",                 // unique key (label-prefixed in compare)
      "session_id": "XNNPACK/mv2/default",       // <- ties record into session
      "region_stack": ["quantization"],          // *no* archive prefix anymore
      "views": {...}, "badges": [...], ... }
  ],

  "dashboard": {                                 // NEW shape: lens -> session_id -> blocks
    "metadata": { "XNNPACK/mv2/default": { "blocks": [...] }, ... },
    "adb":      { "Qualcomm/mobilenet_v2/default": { "blocks": [...] } }
  },

  "graph_assets": {...}, "graph_layers": {...}, "analysis_results": {...},
  "resources": {...}
}
```

**Removed (do not emit and do not read):** `session.start_data`, `session.end_data` (flat archive-wide mirrors), `compare_archives` (replaced by `archives`).

## 4. Archive (raw JSON, `--output-archive`)

```jsonc
{
  "records": [ /* RecordDigest */ ],
  "sessions": [
    { "id", "name", "archive", "start_ts", "end_ts", "start_data", "end_data" }
  ]
}
```

There is no flat `start_data` / `end_data` in the archive either. `Observatory._load_archive_sessions` reads both the new shape and the legacy nested `session: {sessions, ...}` shape (forward-compat for previously-written archives) and synthesises an `archive` field for legacy entries.

## 5. Lens contract

```python
class Frontend:
    def dashboard(
        self,
        session: "Session",                     # the active session
        session_records: List["RecordDigest"], # records whose .session_id == session.id
        analysis: "AnalysisResult",            # archive-scoped analysis for this lens
    ) -> Optional[ViewList]:
        ...
```

The framework calls `dashboard()` once per `(Session, lens)` pair. A report with N sessions and L lenses produces up to N×L invocations. Lenses stay simple — they receive one Session and its records, return blocks; the framework owns the iteration. **There is no `**kwargs` shim** and no flat `start` / `end` / `records` positional view.

`session_records` is filtered before the call: `[r for r in records if r.session_id == session.id]`.

## 6. Compare-mode merge rules

`Observatory.compare_archives(paths, labels, ...)`:

| Field | Rule |
| --- | --- |
| `record.name` | Prefixed `<label>/<name>` for cross-archive uniqueness; collisions get `#2`, `#3` suffixes. |
| `record.session_id` | Prefixed `<label>/<id>`. |
| `record.region_stack` | **Unchanged.** Archive is now its own dimension; tree must not double-encode it. |
| `record.data["graph"]["graph_ref"]` | Rewritten to the prefixed record name so graph viewer asset lookup succeeds. |
| `record.data["per_layer_accuracy"]["graph_ref"]` | Same rewrite as `graph` (was previously missing). |
| `Session.id`, `Session.name` | Prefixed `<label>/<id>`. |
| `Session.archive` | Set to the archive's label. |

## 7. Frontend rendering

| Mode | Sidebar |
| --- | --- |
| Single archive, 1 session | Per-session "📊 Session Dashboard: <name>" link, then records grouped by region tree (or flat). Visually almost identical to the old "📊 Run Dashboard" UI, just renamed. |
| Single archive, N sessions | One section per session in a vertical list. |
| Compare mode (≥ 2 archives) | CSS-grid N-column layout (`grid-template-columns: repeat(--archive-cols, 1fr)`); each column renders its own archive header + sessions + records. |

The dashboard renderer reads:
* `state.data.sessions[].id` to find the active session.
* `state.data.dashboard[lens][sessionId]` to find that session's blocks.
* `state.data.records.filter(r => r.session_id === sessionId)` to derive the session's records.

## 8. Test coverage

The initial commit ships with three test files (78 tests total in the observatory test suite, all green):

| File | What it pins down |
| --- | --- |
| `tests/test_session_first_class.py` (15 tests) | `Session.archive` defaults; `SessionResult` shape; `Frontend.dashboard` signature; per-session payload routing; payload top-level keys; `dashboard[lens][session_id]` shape; per-`(session, lens)` invocation count; `export_json` shape; `_load_archive_sessions` reads new + legacy; `compare_archives` archive grouping; no archive prefix on `region_stack`; graph_ref + per_layer_accuracy graph_ref rewritten; full payload survives JSON encoding. |
| `tests/test_session_dashboards.py` (9 tests) | New dashboards on `accuracy` (run-wide metric means, internal-key exclusion) and `pipeline_graph_collector` (records per innermost region). |
| `tests/test_aot_region_close_and_compare_fixes.py` (4 tests) | `PipelineGraphCollectorLens.close_aot_regions` semantics; `to_executorch` patch closes AOT before runtime work; `device` region lands as a session-root sibling, not under `edge`; per-layer `graph_ref` rewrite. |

## 9. Migration

| Caller | Required change |
| --- | --- |
| Lens `Frontend.dashboard` overrides | Change signature to `(session, session_records, analysis)`. Read from `session.start_data[lens_name]` / `session.end_data[lens_name]`. Drop any `**_kw` shim. Migrated in this commit: `metadata`, `adb`. |
| Custom JS dashboard callbacks | `context` is now `{ session, records }` (was `{ start, end, records }`). Read `context.session.start_data[lensName]` / `context.session.end_data[lensName]`. |
| External readers of `archive.session.start_data` flat dict | Iterate `archive.sessions[]` instead. |

The lens base class change is breaking by design — the old positional signature carried legacy compromises and is not worth preserving for an RFC-stage release.

## 10. Out of scope

* Per-session record selection / record-level compare across sessions inside the same archive.
* "Session diff" view (overlay two Sessions' records side by side). The current compare report's record-level diff stays archive-grain.
