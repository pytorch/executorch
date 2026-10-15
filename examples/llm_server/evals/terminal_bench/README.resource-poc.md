# Background-agent experiment

Compare the same agent with delegation disabled (`agent = "inline"`) or one
optional background child (`agent = "resource"`). The parent owns edits and
verification. A child receives one self-contained inspection question, a bounded
output budget and a separate session. A spawn also runs an independent parent
shell command. Children cannot delegate.

## Configuration

Use a batching server with prefix caching, at least two sessions, and
`max_decode_sequences >= 2` (maximum generation batch size). Add to a working model TOML:

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
warms the model and measures prefill, generation at batch sizes 1 and 2, and mixed steps.
It validates predictions at different prompt lengths; calibration must meet a
30% p90 relative-error limit in each execution category. `calibration.json` is
shared by all trials in the run. Calibration is outside agent time and excluded
from evaluation metrics. Use identical model-cost settings for both A/B modes.

## Admission and measurement

The model proposes one independent question and a parent shell action. The
harness checks the child budget, proposal and in-flight request limit. A child
has at most six turns, 16 KiB of history and the configured total output cap.
These checks do not reserve GPU memory; the server enforces session capacity.

The model supplies only the child's expected non-thinking output tokens, within
the configured output cap.
Thinking is estimated as
`ceil(expected_output_tokens * observed_thinking / observed_non_thinking)` over
completed responses. Forecasts are logged; missing thinking or timing estimates
do not block admission. Actual generation enforces the total output cap.

The harness uses the [Scaling Book's generation cost model](https://jax-ml.github.io/scaling-book/inference/)
as its baseline when attention is memory-bandwidth-bound:

$$
T_{\mathrm{step}} \approx \frac{\sum_i KV_i}{BW}
+ \max\left(\frac{2BN}{C},\frac{W}{BW}\right)
$$

Here, $B$ is the actual generation batch size, $N$ the active parameter count,
$W$ the parameter bytes, and $KV_i$ the attended KV-cache bytes for sequence $i$.
$BW$ and $C$ are calibrated as `effective_bandwidth_bytes_s` and
`effective_compute_flops_s`. The in-flight request limit does not enter this equation.
For generation-only steps, per-token latency is approximately $T_{\mathrm{step}}$
and aggregate throughput is $B/T_{\mathrm{step}}$.

Prefill uses uncached token counts and context lengths, retaining both compute and
memory terms for attention. Calibration adds empirical timing corrections for
generation at batch size 2, mixed steps and fixed prefill overhead. These corrections
do not measure physical byte traffic or peak hardware rates.

At a proposal, the harness logs cold prefill cost and solo/paired decode step
latency at the current prompt lengths. It compares predicted and measured
forwards from the engine's recent-step history. These measurements do not
predict tool execution, future tool outputs or task completion time, and do not
control admission. Calibration error describes inference timing; contexts beyond
the calibration range are extrapolations.

## Evaluation

Compare verifier results and time to successful completion first, then parent
waiting, all generated tokens and uncached input. `agent/events.jsonl` records
model action selections, admission reasons, forecasts and predicted/measured
step times; `parent.json` and `child.json` preserve conversations. A run with no
spawn proposals does not validate admission or batching benefits. Reasoning that
exhausts a response cap without an answer or tool call is an error in both modes.

Assignments are limited to 4 KiB; oversized tool output is omitted. Request
metadata stays out of the conversation. Harbor owns the task deadline. Child
read-only behavior is an instruction, not filesystem isolation. This experiment
supports dense attention models and at most one child, without speculative decoding.
