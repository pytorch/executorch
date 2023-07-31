# Modules and Entrypoints

<b> Disclaimer: Please note that at present, we do not offer any backward
compatibility guarantees for the following APIs. While we are committed to
minimizing significant API changes, it is important to understand that we are
currently in an intensive development phase, and as such, we reserve the right
to modify implementation details and top-level API parameters.  We are
constantly striving to enhance our offerings and deliver the best possible
experience to our users. However, during this phase, it is essential to remain
aware that certain adjustments may be necessary to improve functionality,
stability, or meet evolving requirements. </b>

## Export API

At the top level, the export API is defined as follows:

```python
def export(
    m: Union[torch.nn.Module, Callable[..., Any]],
    args: Union[Dict[str, Tuple[Value, ...]], Tuple[Value, ...]],
    constraints: Optional[List[Constraint]] = None,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        m: the `nn.Module` or callable to trace.

        args: Tracing example inputs.

        constraints: A list of constraints on the dynamic arguments specifying
            their possible range of their shapes

    Returns:
        An ExportedProgram containing the traced method.
    """
```

## Exported Artifact
The export call returns a custom export artifact called [Exported
Program](../ir_spec/00_exir.md#exportedprogram).
