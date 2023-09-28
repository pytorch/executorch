# Generating an ETrecord

Make sure to read and understand what an ETrecord is and why we need it for the SDK in the [getting started page](./00_getting_started.md#etrecord)


For a demo on how to generate an ETrecord, please refer to this [notebook](https://www.internalfb.com/intern/anp/view/?id=3799219).

There are two important API's users must be aware of when dealing with ETrecord: `generate_etrecord` and `parse_etrecord`.

---
### Generating an ETrecord

```python
generate_etrecord(
    etrecord_path: str,
    edge_dialect_program: ExirExportedProgram
    executorch_program: Union[ExecutorchProgram, MultiMethodExecutorchProgram],
    export_modules: Optional[
        Dict[
            str, Union[MultiMethodExirExportedProgram, ExirExportedProgram]
        ]
    ] = None,
) -> None:
```

Generates an ETRecord from the given objects and saves it to the given path.
The objects that will be serialized to an ETRecord are all the graph modules present in the export_modules dict, the graph module present in the edge dialect program object,
and also the graph module present in the executorch program object, which is the closest graph module representation of what is eventually run on the device.

In addition to all the graph modules we also serialize the program buffer which the users can provide to the ExecuTorch runtime to run the model.

#### Parameters:
- `etrecord_path` : Path to where the ETRecord file will be saved to.
- `edge_dialect_program`: ExirExportedProgram for this model returned by the call to to_edge()
- `executorch_program`: ExecutorchProgram or MultiMethodExecutorchProgram for this model returned by the call to to_executorch()
- `export_modules`: Dictionary of graph modules with the key being the user provided name and the value is the corresponding exported module. The exported graph modules can be either the output of capture() or to_edge().

#### Returns:
None

---

### Parsing an ETrecord

```python
parse_etrecord(etrecord_path: str)
```

Parses an ETRecord file and returns a ETRecord object that contains the deserialized graph modules, program buffer and debug handle map. In the graph map in the returned ETRecord object if a model with multiple entry points was provided originally by the user during ETRecord generation then each entry point will be stored as a separate graph module in the ETRecord object with the name being the original module name + "/" + the name of the entry point.

#### Parameters:
 - `etrecord_path`: Path to the ETRecord file.

#### Returns:
 - ETRecord object
