# Terminal-Bench (macOS)

Runs Terminal-Bench 2.0 through Harbor and mini-SWE-agent against the local
ExecuTorch LLM server.

## Prerequisites

Prepare a self-contained `.pte`, matching tokenizer/chat template, and a worker
implementing this checkout's [JSONL protocol](../../cpp/worker_loop.h). The
worker must support the export's backend and input/output signatures.

Follow your model's artifact and worker build instructions. MLX requires Apple
silicon, full Xcode, and the Metal toolchain.

## Run

From the repository root:

```bash
cd examples/llm_server/evals/terminal_bench
bash setup.sh
cp ../configs/terminal-bench.example.toml ../configs/terminal-bench.local.toml
# Adapt the example to your model, server module, and artifact paths.
bash run.sh --config ../configs/terminal-bench.local.toml --dry-run
bash run.sh --config ../configs/terminal-bench.local.toml
```

Setup creates separate Harbor and server environments under
`~/.cache/executorch-evals/`. It reuses Docker or installs Colima through Homebrew.
Model export and worker builds are separate. Set `EXECUTORCH_EVAL_CACHE` before
setup and run to relocate environments and results.

Configuration details:

- Set `python` to an environment with
  [`python/requirements.txt`](../../python/requirements.txt) and any dependencies
  required by your server module. Setup installs only the generic server dependencies.
- Paths support `~`, but not shell-variable expansion. Prefix relative
  `hf_tokenizer` directories with `./` to distinguish them from Hub IDs.
- Match the server module, tool parser, artifact mode, context limit, and sampling
  settings to your model and worker.
- The server binds to `0.0.0.0:8000` for container access; the API is unauthenticated.
  `agent_host` defaults to `host.lima.internal` for Colima and
  `host.docker.internal` otherwise. Override the address/port for your network.

For cross-turn KV reuse, configure named-session capacity in your worker and set
`session_affinity`. The harness sends it as `x-session-affinity`; sequential trials
share this ID. Verify `reused_prompt_tokens` in `server.log`.

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
