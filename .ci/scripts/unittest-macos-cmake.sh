#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# =============================================================================
# AOTI HANG DIAGNOSIS
#
# Run a single AOTI test that is known to hang on macOS CI.  A background
# watchdog samples the native call stack every 60 s so we can see exactly
# which C/C++ function the thread is blocked in (faulthandler only shows
# Python frames and cannot fire when the GIL is held by native code).
# =============================================================================

export TORCHINDUCTOR_CACHE_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/torchinductor_cache_XXXXXX")"
trap 'rm -rf "${TORCHINDUCTOR_CACHE_DIR}"' EXIT

# Force unbuffered output so every print appears immediately in the CI log.
export PYTHONUNBUFFERED=1

# ---------- instrumented test wrapper ----------
cat > /tmp/aoti_diag.py << 'PYEOF'
"""Run a single AOTI test with step-by-step timing instrumentation."""
import json, os, sys, tempfile, time

def log(msg):
    elapsed = time.time() - _t0
    print(f"[AOTI-DIAG +{elapsed:7.1f}s] {msg}", flush=True)

_t0 = time.time()
log("start")

import torch
log(f"torch {torch.__version__} loaded")

from executorch.examples.models.llama3_2_vision.text_decoder.model import Llama3_2Decoder
log("Llama3_2Decoder imported")

params = {
    "dim": 2048,
    "ffn_dim_multiplier": 1.3,
    "fusion_interval": 2,
    "intermediate_dim": 14336,
    "multiple_of": 1024,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 2,
    "n_special_tokens": 8,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vision_chunk_size": 560,
    "vision_max_num_chunks": 4,
    "vocab_size": 21008,
    "vision_num_cross_attention_layers": 1,
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
    json.dump(params, f, indent=2); f.flush()
    model = Llama3_2Decoder(
        encoder_max_seq_len=6404,
        generate_full_logits=True,
        enable_dynamic_shape=True,
        use_kv_cache=True,
        params=f.name,
        dtype=torch.float32,
    )
log("model constructed")

encoder = model.get_eager_model().eval()
for p in encoder.parameters():
    p.requires_grad_(False)
log("model eval + no_grad")

example_inputs = model.get_example_inputs()
example_kwargs = model.get_example_kwarg_inputs()

# Step 1: torch.export
log("step 1/4: torch.export.export ...")
t = time.time()
with torch.no_grad(), torch.inference_mode():
    ep = torch.export.export(encoder, example_inputs, kwargs=example_kwargs, strict=True)
log(f"step 1/4: torch.export.export done ({time.time()-t:.1f}s)")

# Step 2: aoti_compile_and_package
tmpdir = tempfile.mkdtemp()
pkg_path = os.path.join(tmpdir, "text_decoder.pt2")
log(f"step 2/4: aoti_compile_and_package -> {pkg_path} ...")
t = time.time()
path = torch._inductor.aoti_compile_and_package(ep, package_path=pkg_path)
log(f"step 2/4: aoti_compile_and_package done ({time.time()-t:.1f}s)")

# Step 3: aoti_load_package
log("step 3/4: aoti_load_package ...")
t = time.time()
encoder_aoti = torch._inductor.aoti_load_package(path)
log(f"step 3/4: aoti_load_package done ({time.time()-t:.1f}s)")

# Step 4: inference
log("step 4/4: inference ...")
t = time.time()
y = encoder_aoti(*example_inputs, **example_kwargs)
log(f"step 4/4: inference done ({time.time()-t:.1f}s)")

# Verify
eager_res = encoder.forward(*example_inputs, **example_kwargs)
torch.testing.assert_close(y, eager_res, rtol=1e-4, atol=1e-4)
log("PASS — results match")
PYEOF

# ---------- run with background watchdog ----------
# Start the test
${CONDA_RUN} --no-capture-output python /tmp/aoti_diag.py &
TEST_PID=$!

# Watchdog: every 60s, if the test is still running, sample the native stack.
(
  while kill -0 "$TEST_PID" 2>/dev/null; do
    sleep 60
    if kill -0 "$TEST_PID" 2>/dev/null; then
      echo ""
      echo "===== WATCHDOG: native stack sample ($(date)) ====="
      # sample captures C/C++ call stacks on macOS
      sample "$TEST_PID" 1 2>&1 | head -200 || true
      echo "===== END WATCHDOG ====="
      echo ""
    fi
  done
) &
WATCHDOG_PID=$!

# Wait for test, propagate exit code
wait "$TEST_PID"
EXIT_CODE=$?

# Clean up watchdog
kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true

echo "Test exited with code $EXIT_CODE"
exit $EXIT_CODE
