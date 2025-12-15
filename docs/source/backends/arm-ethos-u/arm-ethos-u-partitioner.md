# Partitioner API

The `EthosUPartitioner` controls what parts of a model is delegated to the Arm Ethos-U backend. Below is a reference of the various functions the partitioner provides:

```python
class EthosUPartitioner(compile_spec: executorch.backends.arm.ethosu.compile_spec.EthosUCompileSpec, additional_checks: Optional[Sequence[torch.fx.passes.operator_support.OperatorSupportBase]] = None) -> None
```
Partitions subgraphs supported by the Arm Ethos-U backend.

Args:
- **compile_spec**: List of CompileSpec objects for Ethos-U backend.
- **additional_checks**: Optional sequence of additional operator support checks.

```python
def EthosUPartitioner.ops_to_not_decompose(self, ep: torch.export.exported_program.ExportedProgram) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.node.Node], bool]]]:
```
Return operators and a filter that should not be decomposed.

Provide a base set of ops to preserve as-is and a predicate that keeps
certain activations whole when surrounded by quantize/dequantize ops in
a quantized graph. This helps downstream TOSA lowering and delegation.

Args:
- **ep (ExportedProgram)**: Program used to infer target-specific policy.

Returns:
- **Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]**:
        A list of op overloads to keep intact, and an optional filter
        function that returns True when an op should not be decomposed.

```python
def EthosUPartitioner.partition(self, exported_program: torch.export.exported_program.ExportedProgram) -> executorch.exir.backend.partitioner.PartitionResult:
```
Partition the program and tag TOSA-compatible subgraphs.

Run the FX capability-based partitioner to propose subgraphs, then
refine tags by removing boundary-only quantize/dequantize nodes and by
rejecting partitions that would lower to no-ops. Emit a detailed report
of rejected nodes and their reasons.

Args:
- **exported_program (ExportedProgram)**: Program to analyze and
        partition.

Returns:
- **PartitionResult**: The input program with nodes tagged for delegation
    and a mapping of partition tags to delegation specs.
