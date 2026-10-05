# Terminal-Bench (macOS)

Runs Terminal-Bench 2.0 through Harbor and mini-SWE-agent against the local
ExecuTorch LLM server.

## Prerequisites

Prepare a self-contained `.pte`, matching tokenizer/chat template, and a worker
implementing this checkout's [JSONL protocol](../../cpp/worker_loop.h). The
worker must support the export's backend and input/output signatures.

For model preparation, see [Qwen3](../../../models/qwen3/README.md) and the
[MLX LLM examples](../../../../backends/mlx/examples/llm/README.md). MLX requires
Apple Silicon, full Xcode, and the Metal toolchain.

## Run

From the repository root:

```bash
cd examples/llm_server/evals/terminal_bench
bash setup.sh
cp ../configs/terminal-bench.example.toml ../configs/terminal-bench.local.toml
# Set artifact paths, context limit, and sampling parameters.
bash run.sh --config ../configs/terminal-bench.local.toml --dry-run
bash run.sh --config ../configs/terminal-bench.local.toml
```

Setup creates separate Harbor and server environments under
`~/.cache/executorch-evals/`. It reuses Docker or installs Colima through Homebrew.
Model export and worker builds are separate. Set `EXECUTORCH_EVAL_CACHE` before
setup and run to relocate environments and results.

Configuration details:

- `python` defaults to setup's server environment. Overrides need
  [`python/requirements.txt`](../../python/requirements.txt).
- Paths support `~`, but not shell-variable expansion. Prefix relative
  `hf_tokenizer` directories with `./` to distinguish them from Hub IDs.
- `max_context` must match the export's capacity. Adapt `no_think` and sampling
  settings to the model and worker.
- The server binds to `0.0.0.0:8000` for container access; the API is unauthenticated.
  `agent_host` defaults to `host.lima.internal` for Colima and
  `host.docker.internal` otherwise. Override the address/port for your network.

`--dry-run` writes resolved commands to `run.json`. Use `--task NAME` to override
the task list and `--output DIR` to select a new results directory. Each run owns
its server process.

## Outputs

| File | Contents |
| --- | --- |
| `run.json`, `source.diff`, `agent.yaml` | Resolved configuration, checkout changes, and agent settings |
| `server.log` | Worker and server logs, generation timing |
| `connection.json`, `connection.log` | Preflight response and Docker/curl diagnostics |
| `harbor.log`, `harbor/` | Evaluation logs, rewards, and trajectories |
| `metrics.json` | Token counts and timings, excluding preflight |

Model revisions, export/build commands, and dependency versions are not captured;
retain them alongside results.

Local smoke test: Qwen3-0.6B BF16, in-graph MLX cache, `qwen3_5_moe_worker` with
`(input_ids, cache_position)` inputs. `fix-git` completed with reward 0.0 and
`RepeatedFormatError`. This verifies harness integration, not task accuracy.
