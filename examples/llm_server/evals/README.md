# LLM server evaluations

Each harness lives in its own directory; model configurations live in `configs/`.

## Terminal-Bench

Prepare a worker, exported model, and [server environment](../python/README.md), then:

```bash
bash examples/llm_server/evals/terminal_bench/setup.sh
cp examples/llm_server/evals/configs/terminal-bench.example.toml examples/llm_server/evals/configs/terminal-bench.local.toml
# Edit the model paths in the local TOML.
bash examples/llm_server/evals/terminal_bench/run.sh --config examples/llm_server/evals/configs/terminal-bench.local.toml
```

Setup installs Docker Engine on Ubuntu or Colima on macOS (requires Homebrew),
reusing Docker when available. It installs Harbor in a separate Python 3.12
environment under `~/.cache/executorch-evals`. Harbor installs mini-SWE-agent in
the task container and downloads Terminal-Bench 2.0 tasks on the first run.
Set `EXECUTORCH_EVAL_CACHE` to change the cache location.

`[server]` entries become server CLI flags. `python` selects the server environment;
optional `module` selects a model-specific launcher. Relative model paths resolve
from the TOML directory. `max_context` must fit the export; the server reserves
`max_output_tokens` within that limit. Local TOML files are ignored by Git.

`[terminal_bench]` selects tasks, attempts, and agent limits. Use `--task NAME`
to override the task list (repeatable), `--output DIR` for a new results directory,
or `--dry-run` to inspect the commands. The default task is a smoke check, not a
full benchmark. Use a model capable of tool calling for meaningful task scores.

One server runs for the evaluation, and Harbor runs tasks sequentially with
anonymous requests. The launcher handles Linux host-gateway routing and macOS
Docker Desktop/Colima routing; set `agent_host` under `[terminal_bench]` for a
custom route. The server and its worker stop when the run finishes or is interrupted.

Results are saved under the evaluation cache: `harbor/` contains Harbor's scores,
agent trajectories, and verifier output; `metrics.json` contains aggregate token
counts and prefill/decode times. Server logs, commands, and configuration are also
saved. Infrastructure errors fail the run; a valid task reward of zero does not.
