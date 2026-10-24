# CUDA SiliconBench experiment tools

Prepared with OpenAI Codex and KernelAgent. These are experimental evaluation tools; the candidate kernels have not passed deployment acceptance.

The intended matrix is Muse Glimmer 30B agent and Gemma4 12B agent, ExecuTorch/vLLM/SGLang/llama.cpp FORCE_CUBLAS ON/OFF, prefix caching ON/OFF, concurrency 1/2/4/8/16. Use stock framework code and public switches. Preserve existing data and use a shared GPU lock. Keep model artifacts and every output outside the Git checkout.

Performance means server output tokens divided by complete workload wall time. Run four rounds, discard the first, average the last three. Record output counts, TTFT and sampled memory separately. HTTP TTFT is not pure GPU prefill. The driver records device memory, not allocator allocated/reserved memory. An ExecuTorch memory excess above 15% requires a component audit.

CUDA memory is sampled with `nvidia-smi memory.used` at one-second intervals. It includes resident weights, activations, KV, workspaces, graphs, pool reservations and driver allocations. Short-lived allocation peaks can be missed. Preserve baseline, ready, performance and correctness phases separately; label the result sampled GPU peak rather than an exact allocator peak. SiliconBench's Apple collector instead takes the maximum `mem_used_gb` from a `metalstat` sidecar; the original NVIDIA runner has no such sidecar.

`prepare.py` builds a second stock llama.cpp configuration with only FORCE_CUBLAS changed, pins public GMRID data and builds 0/5-shot prompts. It expects the existing FORCE_CUBLAS build under `build-stock-cublas`. Correctness scripts are from WindChimeRan/SiliconBench commit 55d695bf21060b79498014b5029505fbeb6e54b9. GMRID has 1,147 source rows and eight labels. One source row (5079, Others) has empty input; the unmodified reference builder excludes it, leaving 1,146 scored fixtures, consistent with the repository's committed reference scores. The input audit records this exclusion. Every fixture response ID is checked before scoring, so missing requests cannot disappear from weighted F1. Runtime errors and unparseable responses are scored as incorrect.

```bash
python prepare.py --root /workspace/my-benchmark --cpp-repo /workspace/llama.cpp --output /workspace/my-benchmark/preparation --python /workspace/my-benchmark/.venv/bin/python
```

The correctness inputs are written to `preparation/correctness`; pass that location to `run_case.py`. The preparation script links the reviewed `correctness/scripts` directory into that input directory. Run each server through your launch wrapper. The wrapper must honor SB_PORT, SB_PREFIX_CACHE=on/off and SB_CPP_BUILD and use the same model/quantization/KV/configuration as the performance run.

```bash
python run_case.py --wrapper /workspace/serve.sh --backend vllm --variant vllm-stock --model gemma4-12b-it --prefix-cache off --client /workspace/benchmark_cuda.py --prompts /workspace/agent_benchmark_prompts.json --correctness-root /workspace/my-benchmark/preparation/correctness --python /workspace/my-benchmark/.venv/bin/python --concurrency 16 --output /workspace/my-benchmark/results/vllm-off-c16 --lock /workspace/my-benchmark/gpu.lock
```

The default is four full correctness rounds for each shot setting, in addition to the four performance rounds. Raw conversations, token IDs, logs and model files remain in the experiment output and must not be committed.

`run_matrix.py` runs the selected cache/concurrency/variant cells sequentially. It can install a new, task-owned Supervisor program with `--install-supervisor NAME`; its configuration must not already exist. The shared GPU lock is held until each server is fully stopped. A failed cell stops the matrix and preserves its files for inspection. Collection completion is separate from final model quality, memory, cache-policy and best-ET acceptance.

The `candidates_w4a16_precision_v4` package is Triton W4A16/BF16, packed INT4 bytes unchanged, FP32 affine reconstruction, final BF16 weight operand and FP32 accumulation. Runtime source files are unchanged from the KA v4 output. The validation script's source inventory omits a task-private historical hash inventory; numerical validation is unchanged. M8/M16 are the current candidate buckets. Optimizations for M1/M2/M4 are still pending. All new M8/M16 candidates failed the measured A100 no-regression gate. They must not replace the A100 implementation. RTX5090 validation and real-model F1/performance acceptance are still required.

Validate numeric behavior before measuring any candidate:

```bash
python -m candidates_w4a16_precision_v4.validate --help
```

Generate the public cache/build launch switches from an existing reviewed wrapper using `python make_wrapper.py --base /workspace/base-serve.sh --output /workspace/serve.sh`. The native llama.cpp OFF build may choose quantized activation kernels during prefill; label it as a native-policy comparison rather than A16-confirmed.

The vendored correctness scripts retain the SiliconBench MIT license in `correctness/LICENSE`.

`benchmark_candidates.py` independently measures immutable v4 schedules on RTX5090, using verified existing native search/configuration records and the same Gemma GGUF packer. Its native references are clearly labeled, and candidate output is checked against the independent FP32-affine/BF16 oracle. It does not deploy a kernel. A parent without a CUDA context holds the shared lock until the GPU child has fully exited. `supervise.py` installs one new task-owned job for an explicit reviewed command.

`export_gemma_memory.py` exports a separate Gemma candidate with a smaller prefill write budget. It preserves context12K, session/decode ceilings16, static buckets and64 selected logits rows. Ring capacity follows window+write-budget−1 in both export and runtime; reducing it independently would be unsafe. GGUF and packer hashes are verified, FP32-affine/final-BF16 precision is retained, and the execution plan and optimized compiler commands are audited. Whole-model performance, GMRID F1 and actual sampled memory must pass before replacing any serving artifact.

`evaluate_et_memory.py` waits for that export, verifies its audit and artifact hashes under the shared GPU lock, and generates an isolated wrapper with only the ET artifact path changed. It collects both the original2048-write and candidate512-write artifacts at concurrency1/16 with prefix cachingON/OFF: eight cells, each with four performance rounds and four0/5-shot correctness rounds. Each case holds the GPU lock through server teardown. These measurements select a performance/memory tradeoff; collection alone does not promote an artifact or prove final acceptance.

`continue_prepared_matrix.py` observes an existing task-owned preparation service, including while its output directory is still pending. After successful preparation it checks the pinned dataset and stock build audit, creates an isolated public-switch wrapper and collects40 stock agent cells. A failed preparation or cell stops the continuation and preserves its records. It never restarts an existing job. Best-ET selection and final acceptance remain pending.

The ET workers require resident capacity of active sessions + prefix entries + one transient clone when caching is enabled. With16 active sessions, a17-resident artifact supports cacheOFF; it cannot start cacheON with even one snapshot. Export18 residents for one prefix entry, or33 for16 entries, and account for the additional dummy column in physical KV allocation. `export_gemma_memory.py --max-kv-seqs 18` provides compatible baseline2048-write and candidate512-write exports; their paired evaluation explicitly verifies this capacity and fixes prefix entries to1. It does not claim that one entry is the best final cache policy. The previous17-resident/16-entry queued combination must be superseded before GPU work.

`export_mg_prefix.py` exports solo/text-only MG with the same GGUF and validated quantization/attention sources, BF16, selected64 logits rows, context12K, and existing static methods. Resident capacity is explicitly sized for prefix snapshots and transient cloning. The export holds the GPU lock through CUDA child teardown and audits every compiled KV manifest, compiler optimization and artifact hash. Prefix-aware server performance, memory and GMRID acceptance remain required.


`candidates_w4a16_expanded_v5` preserves the new KA Triton runtime source and extends numerical schedules toM1/2/4/8/16. Its validation inventory omits the private historical inventory only. All16 new A100 candidate gates failed; A100 keeps the original implementation. No candidate is accepted for deployment. `benchmark_candidates_v5.py` measures10 MG synthetic shape controls (M1/2/4 use the actual complete stockW4A8 invocation as timing denominator) and8 actual Gemma packed-weight controls atM8/16. Synthetic MG weights, search-selected native controls and the actual executable selection are explicitly distinguished. Numerical oracle,4 paired timing rounds and GPU exclusivity are unchanged.

`evaluate_mg_prefix.py` waits for a compatible MG export and pinned correctness preparation, checks capacity, model/source/artifact hashes, then collects an isolated ET server matrix. Its initial4 cells are concurrency1/16 and prefixON/OFF, each with4 performance rounds and4 complete0/5-shot correctness rounds. It uses the same artifact forON/OFF and does not select the best final cache policy automatically. The old17-resident artifact is not used for cacheON.


Cache policy candidates must go beyond the minimally compatible1-entry control. Export33 resident sequences for16 prefix entries. MG can compare a2048-write artifact at16 entries with checkpoint intervals0 and512, using `evaluate_mg_prefix.py --cache-modes on --checkpoint-interval 512`. This retains the same16 active/decode ceilings and existing forwards; it does not claim that checkpointing is faster before measuring it. `evaluate_gemma_policy.py` evaluates compatible33-resident/16-entry Gemma512/256-write candidates with the existing precision/weight contract. The original Gemma launcher exposes no periodic-checkpoint flag. Larger cache capacity and smaller write budgets are measured tradeoffs, not automatic best-policy choices. Each control retains full4-round perf/0-shot/5-shot correctness and sampled-memory collection.

When MG's checkpoint policy is configured, the generated wrapper sends an effective interval of zero for cache OFF and the selected interval for cache ON. A positive interval with zero cache entries is invalid at worker startup. Evaluation provenance records both effective values.

`prepare_gemma_checkpoint.py` exposes the existing shared runtime checkpoint interval in a separate worker and a separate copy of the current raw-streaming launcher. It reuses the audited native libraries and preserves graph defaults, initial KV capacity and raw tool/channel text. Preparation owns the GPU lock and leaves the checkout, original binary and original launcher unchanged. CLI checks do not prove model acceptance. `evaluate_gemma_policy.py --checkpoint-preparation DIR --checkpoint-program NAME --checkpoint-interval 512` waits for that build and measures the selected policy with the same full performance/F1/memory protocol; cache OFF always forwards interval zero.

Both policy evaluators accept `--initial-kv-capacity 1024` to exercise existing offgraph growth while retaining the12K maximum context and16 active/decode ceilings. The original controls explicitly preallocate12288 rows. Batched-sequence storage can start smaller and double as needed, including sliding-window layers until their declared ring capacity. Ring addressing keeps its declared safe wrap limit. Growth temporarily retains old and replacement buffers and invalidates captured graph bindings, so measure peak memory, recapture cost, performance and correctness before selecting the dynamic setting. Runtime session admission can be smaller than the exported ceiling, but the current sequence-layout allocation uses the AOT-declared sequence count; these controls do not claim dynamic physical session allocation.


The context limit and physical allocation are separate controls. `export_mg_prefix.py` and `export_gemma_memory.py --max-context 131072` export an explicit128K upper bound; the evaluators require the same `--max-context` and can retain `--initial-kv-capacity 1024`. An existing12K artifact cannot support128K simply by raising a server argument. `configure_context.py` sets the request limit through stock server switches and records an explicit llama.cpp unified pool via `--cpp-pool-tokens`: total shared pool tokens are distinct from the per-slot context ceiling, so a128K shared pool does not promise16 simultaneous128K sequences. Context changes require new, clearly labeled results. This interface does not implement concurrency-dependent physical KV columns.

Performance-first collection uses run_matrix_24h.py and run_case_24h.py. Every performance cell retains four rounds, discarding the first. Correctness uses one complete 0/5-shot evaluation at concurrency 1 and 16, and a fixed balanced 64-item subset at concurrency 2, 4 and 8. Sampled scores are marked smoke and are not full-dataset F1. bootstrap_cpp_shard.py prepares a stock CUDA 12.8 llama.cpp shard from public pinned weights and hash-verified templates.

Prefix policy probes use evaluate_mg_prefix_24h.py and evaluate_gemma_policy_24h.py with --phases performance. Full accepted correctness is then collected for the selected settings. Correctness uses the server-advertised model ID; the requested comparison model name remains in provenance.

select_et_policies_24h.py selects completed error-free c1/c16 prefix policy probes for each cache mode, preserves all candidates and rejected attempts, and queues the selected full main matrix. Its probe plan contains a user-defined tuning cutoff. Selection is provisional until quality and kernel acceptance are reviewed.
