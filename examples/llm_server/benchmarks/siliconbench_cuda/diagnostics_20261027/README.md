# Current ET scaling and original Marlin diagnostics

These isolated scripts preserve the paused full campaigns and accepted main results.

- `run_gemma_scaling.py`: current Gemma P2048 artifact with byte-identical reused weights and unchanged previously qualified native runtime; unprofiled c8/c16 controls and bounded graph-level Nsight traces for chat/prefix ON and agent/prefix OFF. Raw token length and elapsed time are retained; a TTFT measurement difference is never used by itself to explain task throughput.
- `benchmark_original_marlin.py`: unchanged installed vLLM Marlin operation and unchanged ET Triton kernels, with the same pinned public GGUF codes. Exact ET packing functions are executed from source without loading the export framework. Integer zero rounding and FP16 floating zero support are measured separately and are not mislabeled bit-exact. No weight-code quantizer is run. Numerical screening precedes timing. Kernel success is not ET server qualification.
- `run_marlin_campaign.py`: downloads and hashes the exact public Gemma Q4_0 and MG Q4_K_M GGUFs; adds only an isolated gguf dependency directory, without modifying the preserved vLLM environment.

No KA tuning, new static methods, main result replacement, GitHub deletion, or public server exposure is performed.
