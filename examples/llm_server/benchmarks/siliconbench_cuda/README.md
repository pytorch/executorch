# CUDA SiliconBench experiment tools

Prepared with OpenAI Codex and KernelAgent. These are experimental evaluation tools; the candidate kernels have not passed deployment acceptance.

The intended matrix is Muse Glimmer 30B agent and Gemma4 12B agent, ExecuTorch/vLLM/SGLang/llama.cpp FORCE_CUBLAS ON/OFF, prefix caching ON/OFF, concurrency 1/2/4/8/16. Use stock framework code and public switches. Preserve existing data and use a shared GPU lock. Keep model artifacts and every output outside the Git checkout.

Performance means server output tokens divided by complete workload wall time. Run four rounds, discard the first, average the last three. Record output counts, TTFT and sampled memory separately. HTTP TTFT is not pure GPU prefill. The driver records device memory, not allocator allocated/reserved memory. An ExecuTorch memory excess above 15% requires a component audit.

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
