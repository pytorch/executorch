# Simple RNN — Multi-Subgraph AXON Delegation

Demonstrates **multiple AXON subgraphs** in a single model. The RNN
has Linear layers delegated to the AXON NPU, separated by a recurrent
hidden state update (tanh) that runs on the CPU. This forces the
partitioner to create separate command buffers for each group of
delegatable layers.

## Why multiple subgraphs

The AXON NPU accelerates Linear (FC) layers but cannot execute
recurrent operations. In this RNN, the `tanh` activation on the
hidden state is not TOSA INT-compatible, so the partitioner splits
the model:

```
input (4-dim)  hidden (8-dim)
    |               |
    v               v
 fc_ih (4->8)    fc_hh (8->8)      ← AXON subgraph 1
    |               |
    +------ add ----+
            |
          tanh                      ← CPU (breaks delegation)
            |
    +-------+-------+
    |               |
    v               v
 fc_out (8->2)   h_new (8-dim)     ← AXON subgraph 2
    |
  output (2-dim)
```

Each subgraph has its own compiled command buffer. The delegate
lookup table maps subgraph names to compiled models.

## Recurrent execution

The firmware runs 4 RNN steps, feeding the hidden state output back
as input to the next step. Each step dispatches the AXON subgraphs
and runs tanh on the CPU between them.

## Build and run

Same pattern as `hello_axon` — see its README for prerequisites.

```bash
cd examples/nordic/simple_rnn
./setup_export_env.sh                             # one-time
SDK_EDGE_AI_PATH=~/sdk-edge-ai ./run_export.sh    # export model

# In a new terminal:
source ~/ncs-workspace/nrf-connect-sdk-env.sh
cd <executorch-root>
west build -b nrf54lm20dk/nrf54lm20b/cpuapp examples/nordic/simple_rnn \
    --no-sysbuild -- \
    -DZEPHYR_EXTRA_MODULES="$(pwd);$SDK_EDGE_AI_PATH"
west flash
```

## Expected output

```
Simple RNN - ExecuTorch multi-subgraph AXON delegation
AXON NPU: enabled
Loading model (4516 bytes)...
AxonBackend::init (delegate 0, processed=36 bytes)
  AXON model 'rnn_step_4fcd48193cbf' bound (out: 1x8x1 byte_width=1)
AxonBackend::init (delegate 1, processed=36 bytes)
  AXON model 'rnn_step_7ddecacbd5d9' bound (out: 1x2x1 byte_width=1)
Method loaded
  step 0: out=(-0.334, -0.198) 629 us
  step 1: out=(-0.699, -0.296) 691 us
  step 2: out=(-0.433, -0.251) 688 us
  step 3: out=(-0.919, 0.084) 691 us
=== AXON delegate profile ===
handles bound: 2
total infer cycles: 89897 (8 calls)
avg cycles/call: 11237
Done.
```

Key observations:
- `handles bound: 2` — two separate AXON subgraphs
- `8 calls` — 2 subgraphs x 4 RNN steps
- Subgraph 0 (fc_ih + fc_hh → 8 outputs): 12K cycles/call
- Subgraph 1 (fc_out → 2 outputs): 10K cycles/call
- ~690 us per step total (AXON dispatch + CPU tanh + CPU add)
