# RFC: Model naming consistency in `LlmConfig`

**Status:**  **RFC**

**Author:** Mahesh Madhavan

**Last Update:** 2026-06-10


## Scope

This RFC is scoped to the generic, backend-agnostic config in
`extension/llm/export/config/llm_config.py` — specifically `BaseConfig` and the
`ModelType` enum that backs `base.model_class`.

## Motivation

A single field, `base.model_class` (a `ModelType` enum), currently encodes **two
different things at once**, inconsistently:

* **Architecture/family** — e.g. `llama3_2`.
* **A concrete model id** — e.g. `qwen2_5_0_5b` (the normalized HuggingFace
  name, which also drives auto-download via `HUGGING_FACE_REPO_IDS`).
* `static_llama` - this doesn't fall under either category, it's not a family or a specific model id.


This RFC proposes **segregating the two types into explicit fields** so each
has a single, clear meaning.

## Proposal: two fields

```diff
 @dataclass
 class BaseConfig:
-    model_class: ModelType = ModelType.llama3
+    model_class: Optional[ModelType] = None
+    model_family: Optional[ModelFamily] = None
+    model_id: Optional[str] = None        # concrete upstream HF id
     params: Optional[str] = None
     checkpoint: Optional[str] = None
     ...
```

### `model_family` — the architecture

* An enum of **architectures/versions only**: `llama2`, `llama3`, `llama3_1`,
  `llama3_2`, `llama3_2_vision`, `qwen2_5` etc.
* **Optional.** When provided, this would require the checkpoint and params.

### `model_id` — the concrete-model id

* **Optional** string: the normalized upstream HuggingFace model name
  (`qwen2_5_0_5b`, `phi_4_mini`).

* When `checkpoint` is provided along with the `model_id`, download is skipped.

* When omitted, the `model_family` + `params` + `checkpoint` fully
  determine the export (today's model_class behavior).

### How the three weight/identity inputs relate

| Field | Meaning | Required | Example |
|---|---|---|---|
| `model_family` | which architecture/code path | optional | `llama3_2` |
| `model_id` | which concrete HF model | optional | `qwen2_5_0_5b` |
| `checkpoint` | explicit local weights (overrides `model_id` download) | optional | `/path/consolidated.pth` |
| `params` | architecture dims | user-supplied | `/path/to/1b_config.json` |


### `model_class` - existing field
This will become optional - existing models will remain as-is, for backward compatibility

## How it works — examples

```yaml
# Family + explicit local weights (today's Llama flow)
base:
  model_family: llama3_2
  params: /path/to/1b_config.json
  checkpoint: /path/to/consolidated.00.pth

# Concrete id → auto-download the pinned HF checkpoint
base:
  model_id: qwen2_5_0_5b           # → Qwen/Qwen2.5-0.5B via HUGGING_FACE_REPO_IDS
  params: examples/models/qwen2_5/config/0_5b_config.json

# Concrete id + checkpoint override (use local weights, keep the id label)
base:
  model_id: qwen2_5_0_5b
  checkpoint: /path/to/local.pth   # overrides the download
  params: /path/to/0_5b_config.json
```

The family alone is enough to export (with user `params`/`checkpoint`); `model_id`
adds concrete identity and enables HF auto-download.

## Backward compatibility & migration


* The `model_class` field can stay for backward compatibility, but we could eventually deprecate the `--model` arg.
* `params`, `checkpoint`, and all other `BaseConfig` fields are unchanged.

## Affected code

* `extension/llm/export/config/llm_config.py` — `model_family` (new `ModelFamily` enum) + `model_id` (str)
* `examples/models/llama/export_llama_lib.py` — read `model_family` for the
  code-path dispatch (`:1599`, `:674`) and `model_id` for the
  `HUGGING_FACE_REPO_IDS` lookup (`:672`).

## What we're asking for

1. **Agreement** on splitting `model_class` into `model_family` (architecture) +
   `model_id` (concrete HF id).

## Open questions
* **Decision on `static_llama`** — under this split it is neither a family nor a
 model_id. It seems like this functionally behaves the same as llama3_2. Where should this live?

* Is the migration cost of a split justified, over just having an overloaded `ModelType`, and supporting both `model_family` and `model_id` through the existing `model_class`(`ModelType`)? 
