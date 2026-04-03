"""Profile Qwen3.5 MoE decode-phase inference with torch.profiler.

Loads the prequantized model in eager mode (no torch.compile),
runs prefill + warmup decode steps, then profiles ~16 decode steps
and prints the top 30 CUDA operators by total CUDA time.
"""

import torch
from torch.profiler import profile, ProfilerActivity

# Register Triton kernels (fused MoE, GatedDeltaNet)
import executorch.backends.cuda.triton.kernels  # noqa: F401

from executorch.examples.models.qwen3_5_moe.export import load_prequantized_model
from executorch.examples.models.qwen3_5_moe.inference import _move_to_cuda

PREQUANTIZED_DIR = "/home/gasoonjia/models/Qwen3.5-35B-A3B-HQQ-INT4-local/"
PROMPT_TOKENS = [151644, 8948, 198]  # A few token IDs to use as prompt (short)
NUM_WARMUP_DECODE = 3
NUM_PROFILE_DECODE = 16

print("Loading model...")
model, config = load_prequantized_model(PREQUANTIZED_DIR, max_seq_len=4096)
_move_to_cuda(model, config)
model.eval()
print("Model loaded and moved to CUDA.")

# --- Prefill phase (not profiled) ---
print(f"Running prefill with {len(PROMPT_TOKENS)} tokens...")
with torch.no_grad():
    for i, tok_id in enumerate(PROMPT_TOKENS):
        tok = torch.tensor([[tok_id]], dtype=torch.long, device="cuda")
        pos = torch.tensor([i], dtype=torch.long, device="cuda")
        logits = model(tok, pos)

# Get first decode token
next_token = logits[:, -1, :].argmax(dim=-1)
seq_len = len(PROMPT_TOKENS)

# --- Warmup decode steps (not profiled) ---
print(f"Running {NUM_WARMUP_DECODE} warmup decode steps...")
with torch.no_grad():
    for i in range(NUM_WARMUP_DECODE):
        pos = torch.tensor([seq_len + i], device="cuda")
        logits = model(next_token.unsqueeze(0), pos)
        next_token = logits[:, -1, :].argmax(dim=-1)

seq_len += NUM_WARMUP_DECODE

# --- Profiled decode steps ---
print(f"Profiling {NUM_PROFILE_DECODE} decode steps...")
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    with torch.no_grad():
        for i in range(NUM_PROFILE_DECODE):
            pos = torch.tensor([seq_len + i], device="cuda")
            logits = model(next_token.unsqueeze(0), pos)
            next_token = logits[:, -1, :].argmax(dim=-1)

torch.cuda.synchronize()

print("\n" + "=" * 120)
print(f"PROFILE RESULTS ({NUM_PROFILE_DECODE} decode steps, eager mode, batch_size=1)")
print("=" * 120)

# Print top 30 ops sorted by CUDA time total
print("\n--- Top 30 operators by CUDA time total ---")
print(
    prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=30
    )
)

# Also print sorted by CPU time for comparison
print("\n--- Top 30 operators by CPU time total ---")
print(
    prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=30
    )
)

# Export chrome trace for detailed analysis
trace_path = "/tmp/qwen35_moe_profile.json"
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace exported to {trace_path}")
print("Open in chrome://tracing or https://ui.perfetto.dev/")
