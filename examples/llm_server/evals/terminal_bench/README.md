# Terminal-Bench (macOS)

Runs Terminal-Bench 2.0 through Harbor and mini-SWE-agent against the local
ExecuTorch LLM server.

## Prerequisites

Prepare a self-contained `.pte`, matching tokenizer/chat template, and a worker
implementing this checkout's [JSONL protocol](../../cpp/worker_loop.h). The
worker must support the export's backend and input/output signatures.

The example targets Muse-Glimmer 30B's
`muse-glimmer-k-quant-17G-128K-text-solo-metal` export from
[Hugging Face](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE).
See [Muse-Glimmer](../../../models/muse-glimmer/README.md) for artifacts and worker
build instructions. MLX requires Apple silicon, full Xcode, and the Metal toolchain.

## Run

From the repository root:

```bash
cd examples/llm_server/evals/terminal_bench
bash setup.sh
cp ../configs/terminal-bench.example.toml ../configs/terminal-bench.local.toml
# Set Python, worker, model, and tokenizer paths.
bash run.sh --config ../configs/terminal-bench.local.toml --dry-run
bash run.sh --config ../configs/terminal-bench.local.toml
```

Setup creates separate Harbor and server environments under
`~/.cache/executorch-evals/`. It reuses Docker or installs Colima through Homebrew.
Model export and worker builds are separate. Set `EXECUTORCH_EVAL_CACHE` before
setup and run to relocate environments and results.

Configuration details:

- Set `python` to your Muse-Glimmer model environment with PyTorch and
  [`python/requirements.txt`](../../python/requirements.txt) installed. Setup's
  generic server environment does not install the model's dependencies.
- Paths support `~`, but not shell-variable expansion. Prefix relative
  `hf_tokenizer` directories with `./` to distinguish them from Hub IDs.
- The example uses 128K context, ATEM tool parsing, and temperature 0.8. Adapt
  the module, context, and sampling settings when using another model.
- The server binds to `0.0.0.0:8000` for container access; the API is unauthenticated.
  `agent_host` defaults to `host.lima.internal` for Colima and
  `host.docker.internal` otherwise. Override the address/port for your network.

The example enables cross-turn KV reuse with `max_sessions = 2` (one scratch,
one named session) and `session_affinity`. The harness sends the latter as
`x-session-affinity`; sequential trials share this ID. Verify
`reused_prompt_tokens` in `server.log`.

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

Local smoke test with this Muse-Glimmer configuration: `fix-git` completed with
reward 1.0 in 39 turns, with cross-turn KV reuse. This is a single trial.
