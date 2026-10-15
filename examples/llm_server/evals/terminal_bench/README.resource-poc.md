# Background-agent experiment

Compare the same agent with delegation disabled (`agent = "inline"`) or one
optional background child (`agent = "resource"`). The parent owns edits and
verification. A child receives one self-contained inspection question, a bounded
output budget and a separate session. A spawn also runs an independent parent
shell command. Children cannot delegate.

## Configuration

Use a batching server with prefix caching, capacity for two sessions and at least
two decode sequences. Add to a working model TOML:

```toml
[terminal_bench]
agent = "resource" # "inline" for the baseline
concurrency = 1
child_output_tokens = 1536
model_cost_path = "/path/to/model-cost.json"
```

The model-cost JSON describes the actual export, not its advertised model size:

| Field | Meaning |
|---|---|
| `weight_bytes` | Forward weights, including quantization metadata; exclude the embedding lookup table |
| `linear_parameters` | Parameters used in transformer-body matrix multiplies |
| `output_parameters` | Parameters in the output projection, evaluated at selected positions |
| `attention` | List of groups with `layers`, `query_heads`, `kv_heads`, `head_dim`, `kv_bytes` and `window` (0 for global attention) |

Run `bash run.sh --config /path/to/model.toml`. Before Harbor starts, the launcher
warms the model and measures solo, paired and mixed prefill/decode execution.
It validates predictions at different prompt lengths; calibration must meet a
30% p90 relative-error limit in each execution category. `calibration.json` is
shared by all trials in the run. Calibration is outside agent time and excluded
from evaluation metrics. Use identical model-cost settings for both A/B modes.

## Admission

The model forecasts non-thinking output for the child, independent parent work,
and the same remaining work done inline. Thinking is estimated as
`ceil(expected_output_tokens * observed_thinking / observed_non_thinking)` over
completed parent and child responses. Capped or interrupted responses do not
count as completed work. The output cap covers thinking and visible output.

The harness prices these forecasts using a calibrated
[roofline model](https://jax-ml.github.io/scaling-book/inference/): prefill depends
on uncached tokens and retained context; decoding depends on weight sharing,
compute and each active context. It accounts for mixed steps, paired decoding,
the solo tail, tools and integration. Parent and child contexts remain separate.
Prompt counts come from the server's renderer. Unknown costs or savings smaller
than the calibration error mean inline. The model does not choose hardware rates.

Future tool work and output remain estimates. Tool durations and sizes use
observed averages; the two-request approximation places each side's forecast
tool work before its input. Calibration error measures inference timing error,
not uncertainty in the model's work forecast. Contexts outside calibration are
extrapolations. Override `estimate_delegation` or `estimate_thinking_tokens` to
change these policies.

## Evaluation

Compare verifier results and time to successful completion first, then parent
waiting, all generated tokens and uncached input. `agent/events.jsonl` records
forecasts, admission decisions, requests and tools; `parent.json` and `child.json`
preserve conversations. A run with no spawn proposals does not validate admission
or batching benefits. Reasoning that exhausts a response cap without an answer
or tool call is an error in both modes.

Assignments are limited to 4 KiB and child history to 16 KiB; oversized tool
output is omitted. Request metadata stays out of the conversation. Harbor owns
the task deadline. Session/request checks do not reserve GPU memory; child
read-only behavior is an instruction, not filesystem isolation. This experiment
supports dense attention models and at most one child, without speculative decoding.
