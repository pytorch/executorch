# Nordic AXON NPU Examples

Examples demonstrating ExecuTorch deployment on Nordic Semiconductor's
AXON NPU (nRF54LM20B). Each example builds progressively on the
previous one.

## Examples

| Example | What it demonstrates | AXON subgraphs |
|---------|---------------------|----------------|
| [hello_axon](hello_axon/) | Basic AXON delegation: single FC model, export, build, flash | 1 |
| [multi_layer](multi_layer/) | Layer chaining: AXON compiler combines multiple layers into one command buffer | 1 |
| [simple_rnn](simple_rnn/) | Multi-subgraph delegation: FC layers separated by CPU ops (tanh) produce separate command buffers | 2 |

**Start with `hello_axon`** — it has the most detailed README with
setup instructions, Python environment explanation, and step-by-step
walkthrough.

## Prerequisites

- **nRF54LM20DK** development kit
- **nRF Connect SDK (NCS)** with Zephyr
- **Nordic sdk-edge-ai** — set `SDK_EDGE_AI_PATH`
- **uv** — Python package manager (`pip install uv`)

## General workflow

Each example follows the same pattern:

```bash
cd examples/nordic/<example>

# 1. Set up Python export environment (one-time)
./setup_export_env.sh

# 2. Export model (trains, quantizes, compiles to AXON command buffers)
SDK_EDGE_AI_PATH=~/sdk-edge-ai ./run_export.sh

# 3. Build firmware (in a new terminal with NCS toolchain)
source ~/ncs-workspace/nrf-connect-sdk-env.sh
cd <executorch-root>
west build -b nrf54lm20dk/nrf54lm20b/cpuapp examples/nordic/<example> \
    --no-sysbuild -- \
    -DZEPHYR_EXTRA_MODULES="$(pwd);$SDK_EDGE_AI_PATH"

# 4. Flash and check serial output
west flash
```
