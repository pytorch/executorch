# LLM server evaluations

## Terminal-Bench

Prepare a worker, exported model, and [server environment](../python/README.md).
From the repository root:

```bash
cd examples/llm_server/evals
bash terminal_bench/setup.sh
cp configs/terminal-bench.example.toml configs/terminal-bench.local.toml
# Edit the model paths and context limit in the local TOML.
bash terminal_bench/run.sh --config configs/terminal-bench.local.toml
```

Setup installs Docker on Ubuntu or Colima on macOS (requires Homebrew), plus Harbor.
Harbor downloads Terminal-Bench 2.0 tasks and runs mini-SWE-agent in containers.

The TOML selects the model, tasks, attempts, and token budgets. `max_context` must
fit the exported model. The default `fix-git` task is a smoke check; use a model
capable of tool calling for meaningful scores.

Scores, trajectories, logs, and token/timing metrics are saved under
`~/.cache/executorch-evals/terminal-bench/runs/`. Use `--output DIR` to choose a
results directory or `--dry-run` to inspect commands before running.
