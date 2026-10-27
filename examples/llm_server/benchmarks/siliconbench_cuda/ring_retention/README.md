# Ring headroom and allocator retention: mini qualification

The existing full benchmark campaign is paused by user request. New work must
qualify on the existing real-GGUF four-layer MG and six-layer Gemma mini models
before a complete model export or benchmark is started.

The required four conditions are baseline, split-history/current ring only,
allocator retention only, and both. Preserve weights, BF16 KV, logical context
128K, initial capacity512, actual prefill budgets MG4096/Gemma2048, mixed batching,
prefix behavior, and static graph correctness. Measure four rounds, discard the
first, and retain the other three mean and sample SD. Record actual allocation
used/reserved separately from history ring storage, current activation K/V,
phase peaks, and NVML. No main performance result is replaced by this work.

`two_source_ring.py` contains the initial sliding attention and commit
candidate. It reads historical ring and packed current K/V in one online
softmax and commits only the final window of each segment afterwards. Padded
rows cannot write into a real owner. This permits persistent window-sized
history without making every owner retain a full prefill chunk of headroom.

`probe_two_source_ring.py` is only a CUDA kernel prerequisite. Its baseline
uses the previous SDPA source plus a combined scatter before attention. It
covers decode, large prefill, mixed lengths, wrapped history and padded rows;
its timings restore identical history outside CUDA-event timing. It does not
qualify native KV ownership, prefix/rewind, dynamic graphs, allocator policy,
ET mini model numerics, HTTP overhead or full model performance. Those remain
required subsequent tests. Do not publish the kernel probe as an ET speedup.

Source hashes are in `source-manifest.json`. No model, benchmark request,
experiment response, log or result is included in this source package.

`run_ring_compilation_screen.py` compares eager with eager and compiled with
compiled, using the same captured-event timing and numerical checks. The journal
kernel receives its original staging storage and explicit K/V offsets; it avoids
overlapping Python tensor views whose functionalization added full-buffer copies.
Generated call bodies are audited after each run. This is a kernel diagnostic;
the real mini model and native four-condition qualification remain required.
