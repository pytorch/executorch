# Resource-aware agent experiment

This compares the same agent with delegation disabled (`agent = "inline"`) or
one optional background child (`agent = "resource"`). The parent owns edits;
the child investigates one question and returns concise findings. Both use the
same tools and inherit the same prompt prefix. The child has a total output
budget, explicit wait/cancel, and a separate serving session.

Admission runs in the harness. Unknown costs default to inline, as do estimates
that show no completion-time benefit. `estimate_delegation(messages, snapshot)`
is the extension point for calibrated workload estimates; its default returns
`None`, so the base agent stays inline. Estimates cover all child requests and
tool work. The admitted output cap is enforced across the child's responses.

Use the batching server with prefix caching enabled, capacity for two sessions,
and at least two decode sequences. In a working model TOML:

```toml
[terminal_bench]
agent = "resource" # "inline" for the baseline
child_output_tokens = 1536
concurrency = 1
```

Keep the model, server limits, tasks, sampling and output limits identical across
modes. Run through `bash run.sh --config /path/to/model.toml`.

`GET /v1/runtime/stats` reports submitted generations, requests waiting for a
runtime session lease, and the last 16 completed worker measurements. Submitted
work includes the engine queue; these counts do not measure GPU utilization or
decode width. Rates describe individual requests under their observed load and
context, and include their age. Unknown rates remain unknown.

The model receives only the admitted child budget and reason. The harness
rechecks costs and request capacity at launch, leaving room for parent work.
Recent server rates alone cannot predict child work or time saved. Child shell
commands are instructed to be read-only; this is not an OS-enforced filesystem
boundary.

Compare verifier results and agent completion time first, then parent waiting,
all generated tokens and uncached input (`input - cached`). `agent/events.jsonl`
records requests, resource reports, tool execution and delegation; `parent.json`
and `child.json` preserve conversations. A task with no delegation is a valid
result. Two tasks establish feasibility, not a statistically reliable speedup.
