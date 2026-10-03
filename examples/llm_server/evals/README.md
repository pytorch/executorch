# LLM server evaluations

Run end-to-end evaluations through the ExecuTorch LLM server. Terminal-Bench is
available today; additional harnesses such as tau2 and GuideLLM can be added as
sibling directories with their own dependencies and metrics. TOML configurations
live in `configs/`.

## Setup

From the repository root:

```bash
bash examples/llm_server/evals/setup.sh terminal-bench
```

Setup reuses a working Docker installation, or installs Docker Engine on Ubuntu
and Colima with the Docker CLI on macOS. Installing system packages can require
administrator authentication; macOS installation requires Homebrew. On a
provisioned CI runner, add `--ci` to require existing Docker without changing
system services.

The script creates an isolated Python 3.12 environment with Harbor 0.22.0 and
downloads a pinned Terminal-Bench task subset. Harbor installs mini-SWE-agent
2.4.6 inside each task container when the trial starts. No host installation of
the agent is needed. Setup can be rerun; it preserves an unchanged task cache
and reports modifications instead of overwriting them.

Dependencies and tasks live under `~/.cache/executorch-evals` (or
`$XDG_CACHE_HOME/executorch-evals`). Set `EXECUTORCH_EVAL_CACHE` consistently for
setup and run to use another location. Setup does not replace your ExecuTorch
Python environment. Prepare an exported model, compatible native worker, and
[server environment](../python/README.md)
before running an evaluation.

## Configure once

```bash
cp examples/llm_server/evals/configs/terminal-bench.example.toml examples/llm_server/evals/configs/terminal-bench.local.toml
```

Files named `*.local.toml` are ignored by Git so machine-specific paths stay local.
Edit the worker, model, tokenizer, and server Python paths. Set `max_context` to
a capacity supported by the export and choose `max_output_tokens` independently.
The example uses a 2,048-token context for an integration check; use a qualified
model and appropriate context/output budgets for task-quality evaluation.

## Run

```bash
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --suite smoke
```

The command checks prerequisites, starts a fresh server for each trial, probes
the server from a Docker container, runs Harbor, and cleans up its processes.
Linux automatically maps `host.docker.internal` to the host gateway for Harbor
task containers. The launcher uses `host.lima.internal` for a Colima context on
macOS and `host.docker.internal` for Docker Desktop. Use `agent_base_url` for a
custom container route; the server must listen on an address containers can reach.

`smoke` runs `fix-git`; `nightly` adds `openssl-selfsigned-cert`. Both are pinned
subsets, not a full Terminal-Bench score. Use `--task /path/to/task` for a custom
task, or `--task-root /path/to/tasks` with a named suite.

```bash
# Inspect dependencies without running a trial.
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --check
# Record the configuration and commands without launching processes.
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --dry-run
# Run the same subset used by periodic CI.
bash examples/llm_server/evals/run.sh terminal-bench --config examples/llm_server/evals/configs/terminal-bench.local.toml --suite nightly
```

`--check` checks host prerequisites. The container route is tested after the
server starts during an actual run; a particular task's custom network can still
require additional configuration. See the [Terminal-Bench guide](terminal_bench/README.md)
for budget controls, launcher configuration, and resuming runs.

## Results

The command prints a fresh results directory for every invocation, under
`<evaluation-cache>/terminal-bench/runs` unless `output_root` is configured. It contains
`summary.md`, `results.json`, `results.tsv`, resolved configuration, source and
asset checksums, exact commands, server logs, agent trajectories, and verifier
outputs. Setup dependency versions and the selected task revision are recorded.

Results distinguish task rewards from infrastructure errors and report executed
tool observations, token usage, and prefill/decode timings. Reward zero with no
tool use does not establish a meaningful agent rollout.

## Adding an evaluation

Add a sibling package with its own `__main__.py`, `setup.sh`, `run.sh`,
requirements, documentation, and tests; place its TOML examples in `configs/` and register its entry points
in the two top-level scripts. Keep harness-specific behavior and dependencies inside that package.
Extract shared utilities when a second integration needs them. Server protocol
tests remain beside the server implementation in `../python/tests`.

## Periodic CI on Linux GPU and macOS

The [LLM Server Evaluations workflow](../../../.github/workflows/llm-server-evals.yml)
runs the nightly subset at 06:17 UTC and supports manual dispatch. It becomes
active when the repository variable `LLM_SERVER_EVAL_TARGETS` is configured.
Each matrix entry selects its own runner, model configuration, and preparation
script. For example, after provisioning runners with these labels:

```json
[
  {
    "name": "linux-gpu",
    "runner": ["self-hosted", "Linux", "X64", "executorch-evals-gpu"],
    "config": "examples/llm_server/evals/configs/linux-gpu.local.toml",
    "prepare": "/opt/executorch-evals/prepare.sh"
  },
  {
    "name": "macos-mlx",
    "runner": ["self-hosted", "macOS", "ARM64", "executorch-evals-mlx"],
    "config": "examples/llm_server/evals/configs/macos-mlx.local.toml",
    "prepare": "/Users/runner/executorch-evals/prepare.sh"
  }
]
```

Both runners need working Docker and Compose accessible by the runner user. The
Linux GPU runner also needs its model backend and drivers. The macOS runner
needs its selected model backend (for example, Metal/MLX) and Docker Desktop or Colima with virtualization support;
an ordinary hosted macOS VM is not assumed to meet these requirements.

The preparation script receives the current checkout and configuration paths:

```bash
bash /path/to/prepare.sh "$GITHUB_WORKSPACE" "$GITHUB_WORKSPACE/examples/llm_server/evals/configs/macos-mlx.local.toml"
```

CI resolves relative configuration paths against the checkout before calling
the preparation script. It must build the compatible worker from that checkout, prepare the server
environment, obtain pinned model/tokenizer assets, and write or update the
configuration to those paths. Cached weights may be reused; a worker built from
an older checkout would not evaluate the C++ changes under test. Keep model
revision, quantization, context, output allowance, and task settings fixed when
comparing runs. Report Linux and macOS performance separately.

CI invokes the same setup and run scripts as local development. It uploads
configuration, preparation/setup logs, and evaluation artifacts even on failure.
Infrastructure errors fail the job; task rewards are reported without a quality
threshold until a reliable model baseline is established. Lightweight driver
tests continue to run in the existing LLM server CI workflow.
