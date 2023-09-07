# Profiling and debugging delegates

Delegate backends are a prominent component of Edge Models. One attribute of
delegated backends is that they operate mostly as an opaque transformation.
This gives delegate authors greater freedom when defining backend behavior,
but also prevents the Executorch authoring flow from tracking underlying changes.

This makes associating profiling and debug information through delegated
graphs difficult. We have provided a framework that will enable delegate authors
to propagate this information and retrieve it for post run analysis. The process is
broken down into two stages:

1) **Ahead-of-time delegation stage** - Delegate authors need to generate
a debug handle map using the process described below.

2) **Runtime stage** - Delegate authors need to log the profiling data along with the
delegate debug identifiers generated in stage 1 using the API's described below in
the runtime section.

## 1). AOT (ahead-of-time) delegation Stage
### Generating a debug handle map:
**Delegate debug identifiers** are used by delegate authors to mark points of
interest in the lowered graph. Identifiers are associated with operator
nodes of the pre-lowered model graph.

- *For example: If a delegate author wants to signal the fusion of 3 operators
into a single operator of the lowered graph, they would register the 3
original operators to the delegate debug identifier ahead-of-time and then log using the
delegate debug identifier at runtime.*

This is tracked by the `debug_handle_map` and returned as a part of
**PreprocessResult** by the call to `preprocess` from the ahead-of-time implementation of the delegated
backends. The `debug_handle_map` is essentially used as a mechanism to communicate what transformations
occurred in the backend.

```python
class PreprocessResult:
    processed_bytes: bytes = bytes()

    debug_handle_map: Optional[
        Union[Dict[int, Tuple[int]], Dict[int, Tuple[int]]]
    ] = None

    ...
```

The construction of this map is done via a **DelegateMappingBuilder**.


### DelegateMappingBuilder

**DelegateMappingBuilder** is a helper class for managing and constructing
`delegate_handle_map`. A new instance should be used in each `preprocess` call
and the result of this builder should be passed in when constructing
`PreprocessResult`

First, create a DelegateMappingBuilder instance that uses either
manually provided identifiers or generated identifiers for node association.

- `DelegateMappingBuilder()`
  - With __manual identifiers__, users pass in a str or int delegate debug identifier
  when creating entries
- `DelegateMappingBuilder(generated_identifiers=True)`
  - With __generated identifier__, the builder will auto-assign an delegate debug identifier

**Note: A single DelegateMappingBuilder instance can use either manual
or generated identifiers, but not both**


Next, use `insert_delegate_mapping_entry` to iteratively construct the
delegate_map. It takes Node(s) to associate and an optional
delegate debug identifier (only intended to be used for the manual identifiers case described above).
The identifier used is returned from the call.

```python
def insert_delegate_mapping_entry(
    self,
    nodes: Union[fx.Node, List[fx.Node]],
    identifier: Optional[Union[int, str]] = None,
) -> Union[int, str]:
```

Finally, use `get_delegate_mapping` to retrieve the constructed map.
The return value can be directly passed to **PreprocessResults**.

```python
def get_delegate_mapping(
    self,
) -> Union[Dict[int, Tuple[int]], Dict[str, Tuple[int]]]
```

## 2). Runtime stage

NOTE : These API's are not available yet but shown here to give a representation of what the
runtime side of things looks like.

### ID based API's:

If users used integer ID's to generate delegate_debug_identifiers during the AOT process then
they should log their profiling events using the following API's.

Option 1 (For when users can explicitly mark the start and end of an event):
```C++
EventEntry event_entry = EVENT_TRACER_BEGIN_DELEGATE_PROFILING_EVENT_ID(event_tracer, id)
EVENT_TRACER_END_DELEGATE_PROFILING_EVENT_ID(event_entry)
```

Option 2 (For when users only have access to the start and end time of the events after they have occurred.)
```C++
EVENT_TRACER_LOG_DELEGATE_PROFILING_EVENT_ID(event_tracer, id, start_time, end_time)
```

### String based API's:

If users used strings to generate delegate_debug_identifiers during the AOT process then they
should log their profiling events using the following API's.

Option 1 (For when users can explicitly mark the start and end of an event):
```C++
EventEntry = EVENT_TRACER_BEGIN_DELEGATE_PROFILING_EVENT_NAME(event_tracer, name)
EVENT_TRACER_END_DELEGATE_PROFILING_EVENT_NAME(event_entry)
```

Option 2 (For when users only have access to the start and end time of the events after they have occurred.)
```C++
EVENT_TRACER_LOG_DELEGATE_PROFILING_EVENT_NAME(event_tracer, name, start_time, end_time)
```

## Examples:

To indicate how these API's can be used we have provided an end-to-end representative example.

Demo backend that generates delegate mapping for a model that undergoes some simple transformations
in the backend.
`executorch/exir/backend/test/backend_with_delegate_mapping_demo.py`

Corresponding runtime backend code that logs the delegate debug identifiers that were generated
during the ahead-of-time processing done in the above backend example.

`executorch/runtime/executor/test/test_backend_with_delegate_mapping.cpp`
