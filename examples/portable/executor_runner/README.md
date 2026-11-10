# Executor Runner

`executor_runner` loads an ExecuTorch PTE, prepares its inputs, executes the
selected method, and optionally writes tensor outputs for inspection. Build
with `EXECUTORCH_BUILD_EXECUTOR_RUNNER=ON`.

```bash
./executor_runner --model_path=model.pte --method_name=forward
```

Use `--inputs=input0.bin,input1.bin` to supply raw tensor data matching the
method metadata, and `--output_file=result` to save outputs as `result-0.bin`,
`result-1.bin`, and so on. Run `--help` for the available options.

## Server Mode

`--server_mode` keeps the selected method loaded. Each stdin line triggers one
execution; input files are reread before execution. Set `--output_file` to
retrieve the results. `SERVER MODE DONE` acknowledges execution and output
writes. Closing stdin ends the server normally.

## Profiling

Build with `EXECUTORCH_BUILD_DEVTOOLS=ON` and
`EXECUTORCH_ENABLE_EVENT_TRACER=ON`, then pass `--etdump_path=run.etdump`.
Server mode saves the accumulated trace when stdin closes normally, not after
each execution. Terminating the process can lose the trace.

`--num_executions` repeats the supplied inputs; it does not implement
autoregressive generation. Event availability and device-level timing depend
on the linked backend and target driver.
