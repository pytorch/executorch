# Profiling and Debugging Delegates

Delegate backends are a prominent component of Edge Models. One attribute of
delegated backends is that they operate mostly as an opaque transformation.
This gives delegate authors greater freedom when defining backend behavior,
but also prevents the ET authoring flow from tracking underlying changes.

This makes associating profiling and debug information through delegated
graphs difficult. In order to propogate this information, delegate authors
can provide a `debug_handle_map` to communicate what transformations
occured in the backend.

## Debug Handle Map
**Delegate debug identifiers** are used by delegate authors to mark points of
interest in the lowered graph. Identifiers are associated with operator
nodes of the pre-lowered model graph.

- *For example: If a delegate author wants to signal the fusion of 3 operators
into a single operator of the lowered graph, they would register the 3
original operators to the identifier ahead-of-time and then log using the
identifier at runtime.*

This is tracked by the `debug_handle_map` and returned as a part of
**PreprocessResult** from the ahead-of-time implementation of the delegated
backends.

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
`delegate_handle_map`. A new instance should be use in each `preprocess` call
and the result of this builder should be passed in when constructing
`PreprocessResult`

First, create a DelegateMappingBuilder instance that uses either
manually provided identifiers or generated identifiers for node association.

- `DelegateMappingBuilder()`
  - With __manual identifiers__, users pass in a str or int identifier
  when creating entries
- `DelegateMappingBuilder(generated_identifiers=True)`
  - With __generated identifier__, the builder will auto-assign an identifier

**Note: A single DelegateMappingBuilder instance can use either manual
or generated identifiers, but not both**


Next, use `upsert_delegate_mapping_entry` to iteratively construct the
delegate_map. It takes Node(s) to associate and an optional
delegate debug identifier. The identifier used is returned from the call.

- When a new identifier is passed in (or generated), a new entry is created
and associated with the nodes
- When an existing identifier is passed in, the provided nodes are appended
to the existing entry.

```python
def upsert_delegate_mapping_entry(
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

## Runtime APIs
