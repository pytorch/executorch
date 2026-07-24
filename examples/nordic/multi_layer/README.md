# Multi-Layer AXON — Chained Layers

Demonstrates a multi-layer model where the AXON compiler chains
multiple FC layers into a single command buffer. Also showcases
the AXON delegate profiling API.

## Model architecture

```
input (8-dim)
    |
    +----> fc_a (8 -> 16, ReLU) ─┐
    |                             │
    +----> fc_b (8 -> 16, ReLU) ──── multiply ── fc_head (16 -> 4) ── output
```

All operations (FC, ReLU, Multiply) are AXON-supported, so the AXON
compiler chains them into a single command buffer. The entire model
executes in one NPU dispatch call.

## Generated files

After export, `src/generated/` contains:

```
axon_subgraph_multi_layer_<hash>.h      ← command buffer
axon_subgraphs_table.h                  ← lookup: name → compiled model
```

## Expected output

```
Multi-layer AXON — ExecuTorch multi-subgraph delegation
AXON NPU: enabled
Loading model (2084 bytes)...
AxonBackend::init (delegate 0, processed=36 bytes)
  AXON model 'multi_layer_...' bound (out: 1x4x1 byte_width=1)
Method loaded (AXON delegates bound: 0)
  input[0]: class=3 (-27.799, -6.318, -22.745, 30.326) 213 us
  input[1]: class=1 (-26.535, 22.113, -34.117, -14.531) 211 us
  input[2]: class=2 (-22.113, -27.167, 24.008, -17.690) 212 us
  input[3]: class=0 (30.958, -19.586, -15.795, -37.908) 209 us
=== AXON delegate profile ===
handles bound: 1
total infer cycles: 56254 (4 calls)
avg cycles/call: 14063
Done.
```

Note: `handles bound: 1` — all layers fit in one AXON command buffer.
The AXON delegate profiling shows 14K cycles per inference call.

## Build and run

Same pattern as `hello_axon` — see its README for prerequisites.

```bash
cd examples/nordic/multi_layer
./setup_export_env.sh                             # one-time
SDK_EDGE_AI_PATH=~/sdk-edge-ai ./run_export.sh    # export model

# In a new terminal:
source ~/ncs-workspace/nrf-connect-sdk-env.sh
cd <executorch-root>
west build -b nrf54lm20dk/nrf54lm20b/cpuapp examples/nordic/multi_layer \
    --no-sysbuild -- \
    -DZEPHYR_EXTRA_MODULES="$(pwd);$SDK_EDGE_AI_PATH"
west flash
```
