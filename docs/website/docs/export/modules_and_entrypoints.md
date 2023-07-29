<h1> Modules and Entrypoints </h1>

<b> Disclaimer: Please note that at present, we do not offer any backward compatibility guarantees for the following APIs. While we are committed to minimizing significant API changes, it is important to understand that we are currently in an intensive development phase, and as such, we reserve the right to modify implementation details and top-level API parameters.  We are constantly striving to enhance our offerings and deliver the best possible experience to our users. However, during this phase, it is essential to remain aware that certain adjustments may be necessary to improve functionality, stability, or meet evolving requirements. </b>

# Export API

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

# Exported Artifact
The export call returns a custom export artifact called ExportedProgram:

```python
class ExportedProgram:
    graph_module: torch.fx.GraphModule
    graph_signature: ExportGraphSignature
    call_spec: CallSpec
    state_dict: Dict[str, Any]
    symbol_to_range: Dict[sympy.Symbol, Tuple[int, int]]

    @property
    def graph(self):
        return self.graph_module.graph

    def transform(self, *passes: PassType) -> "ExportedProgram":
        # Runs graph based transformations on the given ExportedProgram
        # and returns a new transformed ExportedProgram
        ...

    def add_runtime_assertions(self) -> "ExportedProgram":
        # Adds runtime assertions based on the constraints
        ...

# Information to maintain user calling/returning specs
@dataclasses.dataclass
class CallSpec:
    in_spec: Optional[pytree.TreeSpec] = None
    out_spec: Optional[pytree.TreeSpec] = None


# Extra information for joint graphs
@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: Dict[str, str]
    gradients_to_user_inputs: Dict[str, str]
    loss_output: str


@dataclasses.dataclass
class ExportGraphSignature:
    parameters: List[str]
    buffers: List[str]

    user_inputs: List[str]
    user_outputs: List[str]
    inputs_to_parameters: Dict[str, str]
    inputs_to_buffers: Dict[str, str]

    buffers_to_mutate: Dict[str, str]

    backward_signature: Optional[ExportBackwardSignature]
```
