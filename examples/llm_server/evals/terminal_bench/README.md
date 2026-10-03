# Terminal-Bench (macOS)

Prepare a worker, exported model, and [server environment](../../python/README.md).
Use a worker built for the server checkout; worker protocols can differ between revisions.
From the repository root:

```bash
cd examples/llm_server/evals/terminal_bench
bash setup.sh
cp ../configs/terminal-bench.example.toml ../configs/terminal-bench.local.toml
# Edit the model paths and context limit in the local TOML.
bash run.sh --config ../configs/terminal-bench.local.toml
```

Setup installs Colima (requires Homebrew) and Harbor, reusing Docker if available.
Harbor downloads Terminal-Bench 2.0 tasks and runs mini-SWE-agent in containers.

The TOML selects the model, tasks, attempts, temperature, and token budgets.
`max_context` must fit the exported model. The default `fix-git` task is a smoke
check; use a model capable of tool calling for meaningful scores.

Scores, trajectories, logs, and token/timing metrics are saved under
`~/.cache/executorch-evals/terminal-bench/runs/`. Use `--output DIR` to choose a
results directory or `--dry-run` to inspect commands before running.
Metrics exclude the generation preflight.
