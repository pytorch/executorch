# Delegate Debugging

[Delegate backends](compiler-delegate-and-partitioner.md) are a prominent component of on-device models due to their flexibility in defining behavior. A side effect of this flexibility is that it operates as an opaque transformation. This obfuscates rich associations and mutations that are valuable in post-processing.
- For example, if two different operator fusions were to occur within a delegate, post processing wouldn’t be able to separate the two transformations.

Specifically, it makes associating runtime information (such as profiling results) through delegated graphs difficult. Delegate Debug Identifiers provides a framework through which delegate authors can propagate this information and utilize it for post run analysis.

The preparation is broken down into three stages:
- **Ahead-of-time (AOT)**: Delegate authors generate a __Debug Handle Map__.
- **Runtime**: Delegate authors log using the __Delegate Debug Identifiers__ registered AOT in the __Debug Handle Map__.
- **Deserialization**: Delegate authors provide a parser for custom metadata in delegate events.

## Ahead-of-Time Integration
Delegate authors propagate what transformations occur in a lowered backend by returning a **Debug Handle Map** from the backend implementation.

### Generating a Debug Handle Map
**Debug Handle Maps** communicate what transformations occurred in a backend by mapping **Delegate Debug Identifiers** to debug handles.

**Delegate Debug Identifiers** are generated or user-provided identifiers for representing points of interest during runtime. Recall that debug handles are unique identifiers to operator instances in the model graph.

For example:
- **{ 0: (10, 11), 1: (11, 12) }:** Identifiers 0 and 1 in the runtime correspond to operators with the debug handles (10, 11) and (11, 12) respectively.
- **{ “fused_op_1_2_3”: (11, 12, 15) }**: Identifier “fused_op_1_2_3” in the runtime corresponds to operators with debug handles (11, 12, 15), and 11, 12, 15 corresponds to the op 1, op 2 and op 3.

```{Note}
Identifiers are a means of connecting runtime results to the model graph; the interpretation of the identifiers is defined by the delegate author.
```

**Debug Handle Maps** are constructed through the use of **DelegateMappingBuilder** and returned as a part of `PreprocessResult`.

```python
class PreprocessResult:
    processed_bytes: bytes = bytes()

    debug_handle_map: Optional[
        Union[Dict[int, Tuple[int]], Dict[str, Tuple[int]]]
    ] = None
```
PreprocessResult is defined [here](https://github.com/pytorch/executorch/blob/main/exir/backend/backend_details.py).

#### DelegateMappingBuilder
`DelegateMappingBuilder` is a helper class for managing and constructing Debug Handle Maps. The result of the builder should be passed in when constructing PreprocessResult.

`DelegateMappingBuilder` is defined [here](https://github.com/pytorch/executorch/blob/main/exir/backend/utils.py)

A `DelegateMappingBuilder` instance can be constructed in one of 2 modes: manual identifiers or generated identifiers.

```python
# Manual Identifiers, Default
builder = DelegateMappingBuilder(generated_identifiers=False)

# Generated Identifiers
builder = DelegateMappingBuilder(generated_identifiers=True)
```

With **manual identifiers**, users pass in a **Delegate Debug Identifier** when creating entries.
With **generated identifiers**, the builder will auto-assign a **Delegate Debug Identifier**.

To add an entry to the **Debug Handle Map**, use `insert_delegate_mapping_entry`. It associates one of `fx.Node(s)` or debug handles(s) (sourced from node.meta["debug_handle"]) to an optional **Delegate Debug Identifier** (used for the manual identifiers). The identifier recorded is returned from the call.

```python
def insert_delegate_mapping_entry(
    self,
    nodes: Optional[Union[Node, List[Node]]] = None,
    handles: Optional[Union[int, List[int]]] = None,
    identifier: Optional[Union[int, str]] = None,
) -> Union[int, str]:
```

To retrieve the **Debug Handle Map**, use `get_delegate_mapping`.
```python
def get_delegate_mapping(
    self,
) -> Union[Dict[int, Tuple[int]], Dict[str, Tuple[int]]]
```

A demo of the AOT mapping can be found [here](https://github.com/pytorch/executorch/blob/main/exir/backend/test/backend_with_delegate_mapping_demo.py)


## Runtime Logging
Corresponding to the AOT map, the runtime then defines the functionality through which these events are logged.

### Real-Time Logging

ExecuTorch allows you to log in real time. **Real time Logging** is useful when timestamps are available as the execution occurs. It provides minimal overhead and is intuitive for authors to call.

To log events in real-time (for example, explicitly denoting the profiling start and stop), `event_tracer_start_profiling_delegate` is used to create an `EventEntry` and `event_tracer_end_profiling_delegate` is used to conclude the `EventEntry` for the provided `EventTracer`.

To start an `EventTracerEntry` using `event_tracer_start_profiling_delegate`, the **Delegate Debug Identifier** (provided AOT to the `debug_handle_map`) is passed as either the name or `delegate_debug_id` argument depending on the **Delegate Debug Identifier** type (str and int respectively)

```c++
EventTracerEntry event_tracer_start_profiling_delegate(
    EventTracer* event_tracer,
    const char* name,
    DebugHandle delegate_debug_id)
```

To conclude an `EventTracerEntry`, `event_tracer_end_profiling_delegate` is simply provided the original `EventTracerEntry`.

Optionally, additional runtime `metadata` can also be logged at this point.

```c++
void event_tracer_end_profiling_delegate(
    EventTracer* event_tracer,
    EventTracerEntry event_tracer_entry,
    const void* metadata = nullptr,
    size_t metadata_len = 0)
```

### Post-Time Logging
ExecuTorch also allows you to log in post time. Some runtime settings don't have access to timestamps while it is executing. **Post-Time Logging** enables authors to still be able to log these events.

To log events in post (for example, logging start and end time simultaneously) `event_tracer_log_profiling_delegate` is called with a combination of the arguments used in the real-time logging API’s and timestamps.

```c++
void event_tracer_log_profiling_delegate(
    EventTracer* event_tracer,
    const char* name,
    DebugHandle delegate_debug_id,
    et_timestamp_t start_time,
    et_timestamp_t end_time,
    const void* metadata = nullptr,
    size_t metadata_len = 0)
```
A demo of the runtime code can be found [here](https://github.com/pytorch/executorch/blob/main/runtime/executor/test/test_backend_with_delegate_mapping.cpp).


## Surfacing custom metadata from delegate events

As seen in the runtime logging API's above, users can log an array of bytes along with their delegate profiling event. We make this data available for users in post processing via the [Inspector API](./model-inspector.rst).

Users can pass a metadata parser when creating an instance of the Inspector. The parser is a callable that deserializes the data and returns a list of strings or a dictionary containing key-value pairs. The deserialized data is then added back to the corresponding event in the event block for user consumption. Here's an example of how to write this parser:

NOTE: The input to the deserializer is a list where each entry is a series of bytes (essentially each entry is an immutable bytearray). Users are expected to iterate over this list, deserialize each entry and then return it in the expected format which is either a list of strings, or a dict.

```python
Inspector(
    etdump_path=etdump_path,
    # Optional
    etrecord=etrecord_path,
    # Optional, only needed if debugging was enabled.
    buffer_path=buffer_path,
    delegate_metadata_parser=parse_delegate_metadata
)


def parse_delegate_metadata(delegate_metadatas: List[bytes]) -> Union[List[str], Dict[str, Any]]:
    metadata_str = []
    for metadata_bytes in delegate_metadatas:
        metadata_str += str(metadata_bytes)
    return metadata_str
```
