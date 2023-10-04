# Inspector APIs

## Overview

The Inspector APIs provides a convenient interface for analyzing the contents of [ETRecord](placeholder_link) and [ETDump](placeholder_link), helping developers get insights on model architecture and performance statistics. It's build on top of the `EventBlock` data structure, which organizes a group of `Event`s for easy access to details of profiling events.


There are multiple ways in which users can interact with the Inspector APIs:
- By using public methods provided by the `Inspector` class.
- By accessing the public attributes of the `Inspector`, `EventBlock`, and `Event` classes.
- By using a CLI tool for basic functionalities.

---

## Inspector Methods

### Constructor

Initialize an Inspector instance from the provided ETDump path and an optional ETRecord path.

```python
def __init__(
        self, etdump_path: str, etrecord_path: Optional[str] = None
    ) -> None:
```

#### Parameters:
- `etdump_path` : Path to the ETDump.
- `etrecord_path`: Optional path to the ETRecord.

#### Returns:
- Inspector instance with the underlying `EventBlock`s populated with data from ETDump and ETRecord.

#### Example Usage:
```python
from executorch.sdk.etdb.inspector import Inspector

inspector = Inspector(etdump_path="/path/to/etdump.etdp", etrecord_path="/path/to/etrecord.bin")
```

---

### print_data_tabular

Displays the underlying EventBlocks in a structured tabular format, with each row representing an Event.

```python
def print_data_tabular(self) -> None:
```

#### Parameters:
- None

#### Returns:
- None

#### Example Usage:
```python
inspector.print_data_tabular()
```
```
╒════╤════════════════════╤══════════════════════╤══════════╤══════════╤══════════╤═════════╤═════════╤══════════╤═══════════════════════════════════════════════════╤═══════════════════╤════════════════════════════════╕
│    │   event_block_name │ event_name           │      p50 │      p90 │      avg │     min │     max │   median │ op_types                                          │ is_delegated_op   │ delegate_backend_name          │
╞════╪════════════════════╪══════════════════════╪══════════╪══════════╪══════════╪═════════╪═════════╪══════════╪═══════════════════════════════════════════════════╪═══════════════════╪════════════════════════════════╡
│  0 │                  0 │ Method::init         │ 115.369  │ 115.369  │ 115.369  │ 115.369 │ 115.369 │ 115.369  │ []                                                │ False             │                                │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  1 │                  0 │ Program::load_method │ 193.146  │ 193.146  │ 193.146  │ 193.146 │ 193.146 │ 193.146  │ []                                                │ False             │                                │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  2 │                  1 │ 0                    │   0.001  │   0.001  │   0.001  │   0.001 │   0.001 │   0.001  │ ['aten.convolution.default', 'aten.relu.default'] │ True              │ BackendWithDelegateMappingDemo │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  3 │                  1 │ 1                    │   0.001  │   0.001  │   0.001  │   0.001 │   0.001 │   0.001  │ ['aten.convolution.default', 'aten.relu.default'] │ True              │ BackendWithDelegateMappingDemo │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  4 │                  1 │ 2                    │   0.001  │   0.001  │   0.001  │   0.001 │   0.001 │   0.001  │ ['aten.tan.default']                              │ True              │ BackendWithDelegateMappingDemo │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  5 │                  1 │ 3                    │   0.001  │   0.001  │   0.001  │   0.001 │   0.001 │   0.001  │ ['aten.tan.default']                              │ True              │ BackendWithDelegateMappingDemo │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  6 │                  1 │ 4                    │   0.001  │   0.001  │   0.001  │   0.001 │   0.001 │   0.001  │ ['aten.tan.default']                              │ True              │ BackendWithDelegateMappingDemo │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  7 │                  1 │ DELEGATE_CALL        │  66.1    │  81.3814 │  68.3249 │  54.836 │ 122.845 │  66.1    │ []                                                │ False             │                                │
├────┼────────────────────┼──────────────────────┼──────────┼──────────┼──────────┼─────────┼─────────┼──────────┼───────────────────────────────────────────────────┼───────────────────┼────────────────────────────────┤
│  8 │                  1 │ Method::execute      │  74.5615 │  94.7968 │  79.894  │  59.134 │ 175.748 │  74.5615 │ []                                                │ False             │                                │
╘════╧════════════════════╧══════════════════════╧══════════╧══════════╧══════════╧═════════╧═════════╧══════════╧═══════════════════════════════════════════════════╧═══════════════════╧════════════════════════════════╛
```

---

### find_total_for_module

Returns the total average compute time of all operators within the specified module.

```python
def find_total_for_module(self, module_name: str) -> int:
```

#### Parameters:
- `module_name` : Name of the module to be aggregated against.

#### Returns:
- `int` : Sum of the average compute time (in seconds) of all operators within the module with "module_name".

#### Example Usage:
```python
print(inspector.find_total_for_module("L__self___conv_layer"))
```
```
0.002
```

---

### get_exported_program

Access helper for ETRecord, defaults to returning the Edge Dialect program.

```python
def get_exported_program(self, graph: Optional[str] = None) -> ExportedProgram:
```

#### Parameters:
- `graph` : Optional name of the graph to access. If None, returns the Edge Dialect program.

#### Returns:
- `ExportedProgram` : The ExportedProgram object of "graph".

#### Example Usage:
```python
print(inspector.get_exported_program())
```
```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[4, 3, 64, 64]):
            # No stacktrace found for following nodes
            _param_constant0 = self._param_constant0
            _param_constant1 = self._param_constant1

            ### ... Omit part of the program for documentation readability ... ###

Graph signature: ExportGraphSignature(parameters=[], buffers=[], user_inputs=['arg0_1'], user_outputs=['aten_tan_default'], inputs_to_parameters={}, inputs_to_buffers={}, buffers_to_mutate={}, backward_signature=None, assertion_dep_token=None)
Range constraints: {}
Equality constraints: []
```

---


## Public Attributes

### Event Block Class

Access `EventBlock` instances through the `event_blocks` attribute of an `Inspector` instance, for example:
```python
inspector.event_blocks
```

An EventBlock contains a collection of events associated with a particular profiling/debugging block retrieved from the runtime. Each EventBlock represents a pattern of execution. For example, model initiation and loading lives in a single EventBlock. If there's a control flow, each branch will be represented by a separate EventBlock.

#### Public Attributes:
- `name` : Name of the profiling/debugging block.
- `events` : List of events associated with the profiling/debugging block.

### Event Class

Access `Event` instances through the `events` attribute of an `EventBlock` instance, for example:
```python
for event_block in inspector.event_blocks:
    for event in event_block.events:
        # Do something with each event
```
An Event corresponds to an operator instance with perf data retrieved from the runtime and other metadata from ETRecord.

#### Public Attributes:
- `name` : Name of the profiling/debugging Event.
- `perf_data` : Performance data associated with the event retrived from the runtime (available attributes: p50, p90, avg, min, max and median).
- `op_type` : List of op types corresponding to the event.
- `instruction_id` : Instruction id of the profiling event.
- `delegate_debug_identifier` : Supplemental identifier used in combination with instruction id.
- `debug_handles` : Debug handles in the model graph to which this event is correlated.
- `stack_trace` : A dictionary mapping the name of each associated op to its stack trace.
- `module_hierarchy` : A dictionary mapping the name of each associated op to its module hierarchy.
- `is_delegated_op` : Whether or not the event was delegated.
- `delegate_backend_name` : Name of the backend this event was delegated to.
- `debug_data` : Intermediate data collected during runtime.

#### Example Usage:
```python
for event_block in inspector.event_blocks:
    for event in event_block.events:
        if event.name == "Method::execute":
            print(event.perf_data.raw)
```
```
[175.748, 78.678, 70.429, 122.006, 97.495, 67.603, 70.2, 90.139, 66.344, 64.575, 134.135, 93.85, 74.593, 83.929, 75.859, 73.909, 66.461, 72.102, 84.142, 77.774, 70.038, 80.246, 59.134, 68.496, 67.496, 100.491, 81.162, 74.53, 70.709, 77.112, 59.775, 79.674, 67.54, 79.52, 66.753, 70.425, 71.703, 81.373, 72.306, 72.404, 94.497, 77.588, 79.835, 68.597, 71.237, 88.528, 71.884, 74.047, 81.513, 76.116]
```


---

## CLI

Execute the following command in your terminal to display the data table. This command produces the identical table output as calling the `print_data_tabular`$  API mentioned earlier:

```bash
$ TODO: the CLI is not yet implemented
```

We plan to extend the capabilities of the CLI in the future.
