# Partitioner API

The `VgfPartitioner` controls what parts of a model is delegated to the Arm VGF backend. Below is a reference of the various functions the partitioner provides:

Pass `VgfCompileSpec(alias_buffer_mutations=True)` to delegate mutable KV-cache
buffers as persistent VGF state. Eligible buffers must be zero-initialized,
static-shape `int8` or `float32` tensors whose reads and mutation stay in one
partition. Input and mutation output shapes, dtypes, and layouts must match.
Pass `mutable_buffer_names` to select exact exported buffer names; other mutable
buffers remain program-owned. Omitting it selects all mutable buffers, which
must satisfy the eligibility checks.
This aliases state within one delegate, not across methods or sessions.

```python
class VgfPartitioner(compile_spec: executorch.backends.arm.vgf.compile_spec.VgfCompileSpec, additional_checks: Optional[Sequence[torch.fx.passes.operator_support.OperatorSupportBase]] = None, mutable_buffer_names: Optional[set[str]] = None) -> None
```
Partitions subgraphs supported by the Arm Vgf backend.

Args:
- **compile_spec**: The Vgf compilation specification.
- **additional_checks**: Optional sequence of additional operator support
        checks.
- **mutable_buffer_names**: Exact names to internalize. None selects all
        mutable buffers; unselected buffers remain program-owned.

```python
def VgfPartitioner.ops_to_not_decompose(self, ep: torch.export.exported_program.ExportedProgram) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.node.Node], bool]]]:
```
Preserve only the SDPA nodes that the VGF custom shader can lower.

```python
def VgfPartitioner.partition(self, exported_program: torch.export.exported_program.ExportedProgram) -> executorch.exir.backend.partitioner.PartitionResult:
```
Partition the program and tag TOSA-compatible subgraphs.

Run the FX capability-based partitioner to propose subgraphs, then
refine tags by removing boundary-only quantize/dequantize nodes and by
rejecting partitions that would lower to no-ops. Emit a detailed report
of rejected nodes and their reasons. When VGF aliasing is enabled,
validate and tag eligible mutable buffers for backend ownership.

Args:
- **exported_program (ExportedProgram)**: Program to analyze and
        partition.

Returns:
- **PartitionResult**: The input program with nodes tagged for delegation
    and a mapping of partition tags to delegation specs.

```python
def VgfPartitioner.register_custom_partition_op(self, op: torch._ops.OpOverload) -> None:
```
Register a custom op to be considered supported.

```python
def VgfPartitioner.transform_for_pre_decomposition(self, exported_program: torch.export.exported_program.ExportedProgram) -> torch.export.exported_program.ExportedProgram:
```
Apply required Arm passes before default ATen decompositions.

EXIR invokes this backend extension hook automatically through
``to_edge_transform_and_lower``. Model export users should not call it
directly.

Args:
- **exported_program (ExportedProgram)**: The ATen-dialect program to
        transform.

Returns:
- **ExportedProgram**: The transformed ATen-dialect program.
