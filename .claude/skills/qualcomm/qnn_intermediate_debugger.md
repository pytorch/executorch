# QNN Intermediate Output Debugger — Script Generation

Use this workflow when a user reports a QNN accuracy issue (CPU vs HTP/GPU output divergence) and wants per-layer numerical debugging. The end product is a new Python script — modeled on `examples/qualcomm/util_scripts/qnn_intermediate_debugger_demo.py` — that lowers their model, executes once on device with intermediate dumps enabled, and emits color-coded SVG / CSV diff reports against the edge CPU reference.

This skill **only generates the file**. The user runs it themselves.

---

## When to use

Trigger on user phrases like:
- "QNN accuracy issue / drop / divergence"
- "QNN output doesn't match CPU"
- "debug per-layer / intermediate output for QNN"
- "find which QNN layer is wrong"
- "QcomNumericalComparator / QNNIntermediateDebugger"

Skip and route elsewhere if the user wants:
- *Performance* profiling (optrace / QHAS) → see `profiling.md`
- A new op or quant config → see `new_op_development.md` / `lowering_export.md`
- Final-output-only comparison without per-layer dumps — they don't need this whole pipeline; tell them to compare outputs directly first.

---

## Source of truth

Read these two files in full **before** generating anything. They are the canonical template — do not fabricate API calls.

| File | What it gives you |
|---|---|
| `examples/qualcomm/util_scripts/qnn_intermediate_debugger_demo.py` | End-to-end working example (Inception V3) — copy structure, swap model/dataset |
| `backends/qualcomm/debugger/README.md` (`ExecuTorch QNN Intermediate Output Debugger` section) | API reference, comparator interface, output formats, limitations |

---

## Workflow

### 1. Collect the user's existing example script and run command

Ask the user for **two things**:
1. The path to the example script they currently run for this model. Typically under `examples/qualcomm/scripts/`, `examples/qualcomm/oss_scripts/`, or a private path of their own.
2. The **exact command** they use to run that script — only the user-controlled part: the python invocation and its flags (host, device serial, SoC model, build dir, dataset path, artifact dir, etc.). They do **not** need to include leading environment variables like `QNN_SDK_ROOT=…` or `LD_LIBRARY_PATH=…` — read those from the current shell yourself in step 5.

The command matters because the debug script reuses the same arg parser (`setup_common_args_and_variables`) — at the end you need to hand the user back a working command that runs the new script with the same flags plus `--dump_intermediate_outputs`. Without their original command you can't construct it accurately (you'd have to guess `-H`, `-s`, `-m`, `-b`, `-a`, dataset path, etc.).

Do **not** start writing without both pieces. The generated script is a transformation of theirs, not a from-scratch creation.

If they don't have a script yet, redirect them to `model_enablement.md` first.

### 2. Ask where the generated script should live

The user picks the output path. Do not pick for them. Common choices:
- Same directory as their script with a `_debug.py` suffix
- `examples/qualcomm/util_scripts/` next to the demo
- A scratch path of their choice

### 3. Read the user's script and extract the pieces you need

You need to identify:

| What | Why |
|---|---|
| Model loader (e.g. `MyModel().get_eager_model().eval()`) | Becomes `source_model` in the generated script |
| Sample input (single tensor or tuple) | Passed to `QNNIntermediateDebugger(sample_input=...)` and used for the cosine-similarity sanity check |
| Dataset / calibration inputs | Passed to `build_executorch_binary(dataset=...)` |
| `QnnConfig` setup or args parsing | Reused as-is |
| `pte_filename` / `args.artifact` | Reused; debug artifacts (`etdump.etdp`, `debug_output.bin`) land under the same artifact dir |
| `QuantDtype` (or fp16) | Reused as-is — keep the user's quant choice |
| `SimpleADB` workspace path / device flags | Reused as-is |

If anything is missing or ambiguous in their script (e.g. the model is loaded from a checkpoint and you can't tell what the eager `nn.Module` is), stop and ask.

### 4. Generate the debug script

Mirror the structure of the demo (`qnn_intermediate_debugger_demo.py`). The required transformations against the user's original script:

1. **Imports** — add:
   ```python
   from executorch.backends.qualcomm.debugger.qcom_numerical_comparator_sample import (
       QcomCosineSimilarityComparator,
       QcomMSEComparator,
   )
   from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
       OutputFormat,
       QNNIntermediateDebugger,
   )
   ```

2. **Construct the debugger** before `build_executorch_binary`:
   ```python
   qnn_intermediate_debugger = QNNIntermediateDebugger(sample_input=inputs[0])
   ```

3. **Pass it into `build_executorch_binary`** via `qnn_intermediate_debugger=qnn_intermediate_debugger`. Keep all of the user's other args.

4. **Reduce inference to a single sample** — debug session only supports one execution. Slice the dataset down to `inputs = [inputs[0]]` (and `targets[:1]` if the user uses targets) before `adb.push`.

5. **Define a `validate_intermediate_tensor` callback** that:
   - Calls `qnn_intermediate_debugger.setup_inspector(etdump_path=..., debug_buffer_path=...)`.
   - Runs the edge-CPU module on the sample input and the original `nn.Module` on the same input, then computes a similarity score between the two. **This is the single highest-risk step — read the "Handling model outputs" section below before writing it.** Without this check, per-layer diffs against the edge graph may be misleading.
   - Creates one or more comparators via `qnn_intermediate_debugger.create_comparator(<ComparatorClass>, threshold=...)`. Default to all three: `QcomCosineSimilarityComparator(threshold=0.9)`, `QcomMSEComparator(threshold=0.1)`, and `QcomSQNRComparator(threshold=10.0)` (SQNR is in dB, larger is better) unless the user specifies otherwise.
   - Calls `qnn_intermediate_debugger.generate_results(title=..., path=args.artifact, output_format=OutputFormat.SVG_GRAPH | CSV_FILE, comparator=...)` for each comparator/format combination wanted.

6. **Wire the callback into `adb.pull_debug_output`**:
   ```python
   adb.pull_debug_output(args.artifact, args.artifact, callback=validate_intermediate_tensor)
   ```

7. **Preserve the user's downstream eval logic** (top-k accuracy, IPC client back to a remote, etc.) but it's now running on a single sample — note that in a comment so the user isn't surprised by degenerate metrics.

8. **Assert `dump_intermediate_outputs` at startup** — match the demo's `__main__` block:
   ```python
   assert args.dump_intermediate_outputs, (
       "In order to use intermediate tensor debugger, please provide "
       "the flag --dump_intermediate_outputs when executing."
   )
   ```

### 5. Tell the user how to run it

Construct the run command from the original command they gave you in step 1. **Do not print a generic template** — return the exact command they will copy-paste. Transformation rules:

1. **Swap the script target** — replace the path / module of their original script with the path / module of the generated debug script.
   - If they ran `python -m examples.qualcomm.scripts.foo ...`, change the module to wherever you saved the debug script (e.g. `python -m examples.qualcomm.util_scripts.foo_debug ...`).
   - If they ran `python examples/qualcomm/scripts/foo.py ...`, change the path the same way.
2. **Keep every flag the user had** — `-H`, `-s`, `-m`, `-b`, `-d`, `-a`, any model-specific flags, etc. The debug script reuses `setup_common_args_and_variables`, so they all still apply.
3. **Add `--dump_intermediate_outputs` if it is not already present.** If they already had it, leave it once — don't duplicate.
4. **Auto-detect required env vars from the current shell** — do not ask the user.
   - Check `QNN_SDK_ROOT`, `LD_LIBRARY_PATH`, and `PYTHONPATH` via the Bash tool (`echo $QNN_SDK_ROOT`, etc.).
   - If a variable is already set in the shell, the user's existing process inherits it — do **not** prepend it to the command (it would be redundant and noisy).
   - If a variable is **unset** but is required for the QNN flow (typically `QNN_SDK_ROOT`), prepend it inline only if you can determine a sensible value (e.g. from a previous build invocation in this conversation). Otherwise call it out as a prerequisite the user needs to export themselves rather than hardcoding a guess.
5. **Format on multiple lines with `\` continuations** for readability when the command is long.

Present the result as a fenced bash block, prefixed by a one-line note of what changed vs. their original. Example output:

> Here's the command — same as your original, with the script swapped to the new debug file and `--dump_intermediate_outputs` added:
>
> ```bash
> python -m examples.qualcomm.util_scripts.my_model_debug \
>     -H $HOST -s $DEVICE_SERIAL -m $SOC_MODEL -b build-android \
>     -d /path/to/dataset -a ./my_model_debug \
>     --dump_intermediate_outputs
> ```

If the user did not give you a runnable original command in step 1 (e.g. they pasted only the script path), do **not** fabricate values for `-H` / `-s` / `-m` / `-b` / `-a` / `-d`. Stop and ask before printing — wrong device or SoC values waste a full export + on-device run.

After the command runs, the artifact dir will contain SVG / CSV reports — green nodes pass, red nodes fail the comparator threshold. That's the first place to look for the layer that introduces the gap.

---

## Comparators — defaults and customization

Out-of-the-box (from `qcom_numerical_comparator_sample.py`):
- `QcomCosineSimilarityComparator(threshold=0.9)` — flag if cosine drops below 0.9
- `QcomMSEComparator(threshold=0.1)` — flag if MSE exceeds 0.1
- `QcomSQNRComparator(threshold=10.0)` — flag if SQNR (dB) drops below 10. Backed by `torchao.quantization.utils.compute_error`. Larger is better; 10 dB is a permissive baseline for INT8 quantized graphs — tighten for FP16.

If the user wants something else (e.g. max abs diff, custom logit-space metric), point them at `QcomNumericalComparatorBase` and stub out a derived class. The base handles QNN dequantization + layout transform via `preprocessing` — they only implement `metric_name()`, `is_valid_score()`, and `element_compare()`. Do **not** override `preprocessing`; the base intentionally locks it down.

---

## Handling model outputs (highest-risk part of generation)

The nn.Module-vs-edge sanity check looks innocent in the demo (Inception V3 returns a single tensor) but breaks silently the moment a real model returns anything else. Before writing this block, **inspect what the user's model and edge graph actually return** — don't assume it's a single tensor. The same care applies to both sides; the eager `nn.Module` and `edge_ep.module()` may return different shapes/structures even from the same source model.

### Common output shapes and how to handle them

| Eager model returns | What you must do |
|---|---|
| Single `Tensor` | `out.flatten()` directly. |
| `tuple` / `list` of tensors (e.g. classifier + aux head, encoder hidden states) | Compare **every** element pairwise. Don't pick `[0]` and call it done — the user is debugging accuracy, hidden divergence in the dropped outputs is exactly what they're trying to find. |
| Custom dataclass / `ModelOutput` (HuggingFace style — `BaseModelOutput`, `CausalLMOutputWithPast`, etc.) | Extract the tensor field(s) explicitly (`out.logits`, `out.last_hidden_state`, etc.). Field name varies by model — read the user's model definition or the eager output's `.__dataclass_fields__` to confirm. |
| `dict` of tensors | Iterate over keys; sanity-check both sides have the same key set first. |
| Tensors with different dtypes between eager and edge | Cast to a common dtype (typically `float32`) before similarity. |

### Required generation behavior

1. **Compute eager and edge outputs first**, then branch on their actual structure. Don't hardcode `result.flatten()` — write a small adapter that inspects the type and pulls tensors out.
2. **Compare every output**, not just `[0]`. For a multi-output model, emit one similarity score per output and label them (e.g. `output[0]: cos=0.998`, `logits: cos=0.92`).
3. **Match what the model's `forward` actually accepts.** `qnn_intermediate_debugger.sample_input` is what the debugger was constructed with — verify that calling `source_model(*sample_input)` and `edge_ep.module()(*sample_input)` both work with the user's signature. Some users pass kwargs, some pass a single tensor without unpacking, some pass a tuple. Mirror exactly what their existing script does.
4. **If the eager and edge outputs structurally differ** (e.g. eager returns a dataclass but edge returns a tuple after `torch.export`), normalize both into the same shape (typically a flat list of tensors in declared order) before comparing.
5. **If anything is ambiguous after reading the model, stop and ask the user.** Wrong handling here silently invalidates the entire downstream comparison and is the most likely way for this generated script to mislead the user.

### Sketch (for a single-tensor model)

```python
edge_result = qnn_intermediate_debugger.edge_ep.module()(*qnn_intermediate_debugger.sample_input)
with torch.no_grad():
    source_result = source_model(*qnn_intermediate_debugger.sample_input)
score = torch.nn.functional.cosine_similarity(
    edge_result.flatten().to(torch.float32),
    source_result.flatten().to(torch.float32),
    dim=0,
).item()
print(f"Cosine similarity (nn.Module vs edge CPU): {score:.6f}")
```

### Sketch (for a multi-output / dataclass model — adapt to actual structure)

```python
def _to_tensor_list(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    # Dataclass / ModelOutput — pick the fields the user actually cares about
    return [getattr(out, name) for name in ("logits", "last_hidden_state") if hasattr(out, name)]

edge_tensors = _to_tensor_list(qnn_intermediate_debugger.edge_ep.module()(*qnn_intermediate_debugger.sample_input))
with torch.no_grad():
    source_tensors = _to_tensor_list(source_model(*qnn_intermediate_debugger.sample_input))

assert len(edge_tensors) == len(source_tensors), (
    f"Output count mismatch: edge={len(edge_tensors)} vs eager={len(source_tensors)}"
)
for i, (e, s) in enumerate(zip(edge_tensors, source_tensors)):
    score = torch.nn.functional.cosine_similarity(
        e.flatten().to(torch.float32), s.flatten().to(torch.float32), dim=0
    ).item()
    print(f"Cosine similarity[{i}] (nn.Module vs edge CPU): {score:.6f}")
```

The exact field names in `_to_tensor_list` are placeholders — replace with what the user's model actually returns. If you can't determine it from the script alone, ask.

---

## Hard requirements / limitations to surface to the user

Pulled directly from the README — call these out before they spend time debugging the wrong thing:

1. **One execution per debug session.** Multiple `adb.execute()` calls in a single session produce undefined results. Always reduce dataset to a single sample.
2. **No partial delegation.** If their model has CPU fallbacks, the comparator graph may be incomplete or wrong. Verify full delegation first (see `model_enablement.md` step 3).
3. **No LLM models.**
4. **No multi-method graphs.**
5. **Custom runners must implement etdump.** If the user wrote their own runner instead of using `qnn_executor_runner`, point them at the [etdump tutorial](https://pytorch.org/executorch/stable/etdump.html). Without etdump, no `etdump.etdp` is produced and the inspector has nothing to compare.
6. **`--dump_intermediate_outputs` is required.** Otherwise QNN doesn't dump per-layer tensors and the entire pipeline collapses.

If any of 2–4 apply, tell the user this skill's output won't help them and stop — don't generate a script that will silently produce garbage.

---

## Common pitfalls when generating

- **Forgetting to slice the dataset to one sample** — script will run multiple times, debug output is undefined.
- **Using `inputs[0]` as `sample_input` when `inputs` is a list of tuples** — `QNNIntermediateDebugger(sample_input=...)` expects the same shape that the model's `forward` accepts. Match what the user's existing script passes to `model(*inputs)`.
- **Reusing the user's `dataset=inputs` after slicing** — `build_executorch_binary` wants the *original* (calibration) dataset for quantization; only the post-build inference path is sliced. Slice after `build_executorch_binary`, before `adb.push`.
- **Overriding `preprocessing` on a custom comparator** — base class raises `TypeError` in `__init_subclass__`. Don't try.
- **Skipping the nn.Module-vs-edge cosine check** — per-layer comparisons compare QNN against the edge CPU graph, not against eager. If the edge graph already differs from eager (quant calibration, pass transform), every "failure" downstream may be a red herring. Always include this check.
