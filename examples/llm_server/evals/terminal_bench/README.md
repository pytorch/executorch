# Terminal-Bench with the LLM server

Use the [evaluation quick start](../README.md) to install Docker and Harbor and
download the pinned tasks. With an exported model and configured worker, run:

```bash
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --suite smoke
```

Copy [the example configuration](../configs/terminal-bench.example.toml), fill in
local model/tokenizer/worker paths, and select your budgets. The driver starts
and stops the Python server and worker, invokes Harbor, and records a fresh
results directory for every invocation. It defaults to one attempt per task.

Install the [server dependencies](../../python/requirements.txt) in the interpreter
selected by `server_python`. Setup installs [evaluation dependencies](requirements.txt)
in a separate Python 3.12 environment; `harbor_bin` can select a custom installation.
Use `--suite smoke` or `--suite nightly` for the downloaded task subsets, or
`--task` for a local directory containing `task.toml` and `instruction.md`.
Tasks retain their own container images, verifier, and time limits.

Harbor 0.22.0 installs mini-SWE-agent 2.4.6 in each task container. Model export,
worker builds, and the server environment remain separate from harness setup.

## Configuration and controls

```bash
# Check local dependencies and Docker without starting a trial.
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --check
# Record resolved settings and exact commands without starting processes.
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --dry-run
# Resume an unchanged run, retaining completed trials.
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --run-dir /path/to/run --resume
```

TOML keys match CLI options with underscores. Paths are relative to the config
file; explicit CLI options override the corresponding config value, including
lists. Omitting `run_dir` creates a unique directory beneath `output_root`.
Resume compares settings, source revision/diff, driver, task contents, and
worker/model/tokenizer checksums. Code or asset changes require a fresh run.

| Control | Meaning |
| --- | --- |
| `max_context` | Context limit enforced by the Python server; must fit the worker and export |
| `max_output_tokens` | Per-response generation allowance reserved within that context |
| `step_limit` | Agent iteration cap |
| `attempts` | Independent trials per task, each with a fresh server |
| `session_affinity` | Opt into a named session for the whole task; requires worker support |

Each admitted prompt must satisfy `prompt_tokens + max_output_tokens <= max_context`.
The context flag cannot enlarge the exported model or change the worker's own
capacity. The model's real HF tokenizer is required for templating and counting.

The default launcher is `executorch.examples.llm_server.python.server`. For a
model-specific launcher, set `server_module` and pass additional options through
`server_arg`, a list of `--name=value` strings. It must accept the common worker,
model, tokenizer, host, port, and context flags shown by `--dry-run`. For example,
`server_arg = ["--assistant-header=<|im_start|>assistant\n"]` configures the generic
server's assistant header. Match its tool parser to the model's output format.
Worker-specific settings belong in that model's launcher or an executable wrapper
selected by `worker_bin`; the driver does not assume a cache layout or worker API.

The container-visible `agent_base_url` must point to this server's port and end
in `/v1`. The launcher selects `host.lima.internal` for Colima and
`host.docker.internal` for Docker Desktop. On Linux, the driver supplies a Harbor
Compose overlay mapping `host.docker.internal` to the Docker host gateway.
`--check` validates host prerequisites; actual runs probe `/health` from a
container before starting Harbor. A task with custom networking may need a
different `agent_base_url`. Inspect `container-check.log`, `server.log`, and
`harbor.log` when diagnosing failures.

## Execution and evidence

```text
Harbor / mini-SWE-agent / task container
    → OpenAI HTTP → Python LLM server → worker → model
```

The harness owns task execution, full conversation history, and verification.
Every trial starts a fresh server and worker. With `session_affinity = true`,
it also resets a unique named session before execution and closes it afterward.
Without affinity, it supports workers that only accept anonymous requests.
Processes are cleaned up on completion or interruption. Trials run sequentially.

Each run saves `manifest.json`, `plan.json`, `summary.md`, `results.json`, and
`results.tsv`. Each trial saves exact commands, server/Harbor logs, the agent
trajectory, verifier outputs, and `outcome.json`. Results include task reward,
agent exit reason, observed tool responses, token usage, peak prompt length,
and prefill/decode timings. Startup and task wall time are recorded separately.
Cumulative input tokens measure traffic; peak prompt length measures context
pressure. Compare timings only across like-for-like models and hardware.

Reward zero with `RepeatedFormatError` and no tool observations means the model
did not execute a meaningful task rollout. An earlier Qwen3-0.6B smoke run had
this outcome; it validated connectivity, not task quality. Use a qualified model
and suitable context/output budgets before interpreting accuracy results.
Missing server metrics, context violations, or failed cleanup mark a trial as a
harness error. Infrastructure failures return a nonzero status; valid scored
trials return zero even when their reward is zero.

Driver tests cover configuration, process isolation, interruption, resume,
container routing, metric reporting, and the real Python HTTP server with a
fixture worker. Fixture rewards are not Terminal-Bench scores.
