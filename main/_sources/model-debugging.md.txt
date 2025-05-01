# Debugging Models in ExecuTorch

With the ExecuTorch Developer Tools, users can debug their models for numerical inaccurcies and extract model outputs from their device to do quality analysis (such as Signal-to-Noise, Mean square error etc.).

Currently, ExecuTorch supports the following debugging flows:
- Extraction of model level outputs via ETDump.
- Extraction of intermediate outputs (outside of delegates) via ETDump:
  - Linking of these intermediate outputs back to the eager model python code.


## Steps to debug a model in ExecuTorch

### Runtime
For a real example reflecting the steps below, please refer to [example_runner.cpp](https://github.com/pytorch/executorch/blob/main/examples/devtools/example_runner/example_runner.cpp).

1. [Optional] Generate an [ETRecord](./etrecord.rst) while exporting your model. When provided, this enables users to link profiling information back to the eager model source code (with stack traces and module hierarchy).
2. Integrate [ETDump generation](./etdump.md) into the runtime and set the debugging level by configuring the `ETDumpGen` object. Then, provide an additional buffer to which intermediate outputs and program outputs will be written. Currently we support two levels of debugging:
    - Program level outputs
    ```C++
    Span<uint8_t> buffer((uint8_t*)debug_buffer, debug_buffer_size);
    etdump_gen.set_debug_buffer(buffer);
    etdump_gen.set_event_tracer_debug_level(
        EventTracerDebugLogLevel::kProgramOutputs);
    ```

    - Intermediate outputs of executed (non-delegated) operations (will include the program level outputs too)
    ```C++
    Span<uint8_t> buffer((uint8_t*)debug_buffer, debug_buffer_size);
    etdump_gen.set_debug_buffer(buffer);
    etdump_gen.set_event_tracer_debug_level(
        EventTracerDebugLogLevel::kIntermediateOutputs);
    ```
3. Build the runtime with the pre-processor flag that enables tracking of debug events. Instructions are in the [ETDump documentation](./etdump.md).
4. Run your model and dump out the ETDump buffer as described [here](./etdump.md). (Do so similarly for the debug buffer if configured above)


### Accessing the debug outputs post run using the Inspector API's
Once a model has been run, using the generated ETDump and debug buffers, users can leverage the [Inspector API's](./model-inspector.rst) to inspect these debug outputs.

```python
from executorch.devtools import Inspector

# Create an Inspector instance with etdump and the debug buffer.
inspector = Inspector(etdump_path=etdump_path,
            buffer_path = buffer_path,
            # etrecord is optional, if provided it'll link back
            # the runtime events to the eager model python source code.
            etrecord = etrecord_path)

# Accessing program outputs is as simple as this:
for event_block in inspector.event_blocks:
    if event_block.name == "Execute":
        print(event_blocks.run_output)

# Accessing intermediate outputs from each event (an event here is essentially an instruction that executed in the runtime).
for event_block in inspector.event_blocks:
    if event_block.name == "Execute":
        for event in event_block.events:
            print(event.debug_data)
            # If an ETRecord was provided by the user during Inspector initialization, users
            # can print the stacktraces and module hierarchy of these events.
            print(event.stack_traces)
            print(event.module_hierarchy)
```

We've also provided a simple set of utilities that let users perform quality analysis of their model outputs with respect to a set of reference outputs (possibly from the eager mode model).


```python
from executorch.devtools.inspector import compare_results

# Run a simple quality analysis between the model outputs sourced from the
# runtime and a set of reference outputs.
#
# Setting plot to True will result in the quality metrics being graphed
# and displayed (when run from a notebook) and will be written out to the
# filesystem. A dictionary will always be returned which will contain the
# results.
for event_block in inspector.event_blocks:
    if event_block.name == "Execute":
        compare_results(event_blocks.run_output, ref_outputs, plot = True)
```
