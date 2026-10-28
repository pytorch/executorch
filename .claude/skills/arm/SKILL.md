---
name: arm
description: Develop, debug, or explain the Arm backend in backends/arm/ and its community-facing examples in examples/arm/. Covers partitioning, TOSA lowering, VGF, Ethos-U, quantization, graph passes, memory layout, and node visitors. Not for development of other backends.
---

# Arm Backend Development

Use this guide to choose the implementation layer and validation path, then
read the relevant code for current interfaces. Scope work to `backends/arm/`
and its supporting examples and tests.

## Arm Targets and Code Map

The Arm backend lowers to TOSA IR. Target-specific compilation then produces
VGF using Model Converter or Ethos-U NPU code using Vela. TOSA versions,
profiles, and extensions affect supported operations and lowering choices;
consult `TosaSpecification` in `backends/arm/tosa/specification.py`.
Specification documents are available at <https://www.mlplatform.org/tosa/>.

| Location | Responsibility |
|----------|----------------|
| `backends/arm/quantizer/` | Quantization annotation and configuration |
| `backends/arm/common/arm_compile_spec.py` | Shared Arm compile configuration |
| `backends/arm/tosa/` | TOSA backend, partitioner, and compile spec |
| `backends/arm/tosa/partitioner.py` | Shared capability-based partitioning, delegation tags, and partition-boundary handling |
| `backends/arm/operator_support/` | Operator eligibility and target-specific restrictions; `tosa_supported_operators.py` composes support checks |
| `backends/arm/tosa/specification.py` | TOSA versions, profiles, extensions, and target support |
| `backends/arm/vgf/` | VGF backend, partitioner, and compile spec |
| `backends/arm/vgf/partitioner.py` | VGF specialization of the shared TOSA partitioner |
| `backends/arm/ethosu/` | Ethos-U backend, partitioner, and compile spec |
| `backends/arm/ethosu/partitioner.py` | Ethos-U specialization of the shared TOSA partitioner |
| `backends/arm/_passes/arm_pass_manager.py` | Pass ordering and pipeline configuration |
| `backends/arm/_passes/` | Decompositions, layout handling, dialect conversion, and optimizations |
| `backends/arm/operators/` | Node visitors and TOSA serialization |
| `backends/arm/test/tester/test_pipeline.py` | Reusable lowering, execution, and pass-test pipelines |
| `backends/arm/test/` | Operator, pass, model, and integration tests |
| `examples/arm/` | Community-facing getting-started and evaluation examples; dependency setup |

## Partitioning

Partitioning selects supported regions of the Edge graph for delegation before
backend lowering. `TOSAPartitioner` uses capability-based partitioning to group
eligible nodes and assign `delegation_tag` metadata. Its `DelegationSpec` selects
the backend and compile specs used to lower each region. `VgfPartitioner` and
`EthosUPartitioner` reuse this machinery with target-specific configuration.
Nodes left outside delegates remain in the ExecuTorch graph and need suitable
runtime kernels; partial delegation is not itself a lowering failure.

Operator eligibility is more than the presence of a node visitor. Support checks
in `backends/arm/operator_support/` account for the selected TOSA specification,
target restrictions, dtypes, shapes, quantization, and operator arguments.
`tosa_support_factory` combines positive support checks with rejection checks;
`additional_checks` adds restrictions, not new lowering implementations.

When changing or debugging partitioning:

- Use `WhyNoPartitionReporter` rejection reasons and inspect delegation tags to
  understand why nodes were excluded before broadening support checks.
- Change operator eligibility in the support checks; change region grouping or
  boundary handling in the partitioner. Keep graph rewrites in the appropriate
  transformation passes rather than hiding missing lowering support in checks.
- Preserve Q/DQ boundaries and constant or mutable-buffer ownership when
  changing region boundaries. Acceptance must agree with what the selected
  backend can actually lower and execute.
- Test both accepted and rejected cases, partition boundaries, and fallback
  behavior. Start with `backends/arm/test/misc/test_tosa_operator_support.py`
  for support checks and `backends/arm/test/misc/test_custom_partition.py` for
  partitioning examples; use operator pipelines to verify delegated execution.

## Arm Lowering Pipeline

The pass manager is authoritative for ordering and target-specific differences.
The responsibilities below are not a fixed execution sequence; passes can be
interleaved or repeated:

- Q/DQ folding attaches quantization parameters to nodes and removes folded Q/DQ operators;
- Decompositions rewrite ATen operators toward TOSA forms.
- Memory-layout lowering first wraps channels-last nodes with input and output
  permutes: channels-first to channels-last before the node, and the inverse
  after it. This preserves the surrounding graph's channels-first semantics.
  Later propagation moves permutes through compatible operators, bringing
  inverse pairs together so they cancel instead of transposing at every node.
  Necessary conversions remain at layout boundaries; not all permutes disappear.
- Dialect conversion maps lowered Edge/ATen operators to TOSA equivalents;
  optimizations deduplicate nodes, remove no-ops, and fuse operators.
- Node visitors serialize the lowered graph with `tosa_serializer`, before
  target-specific compilation.

## Implementation Decisions

- Extend nearby implementations, interfaces, and helper methods before adding
  parallel mechanisms. Fix the earliest responsible layer rather than recovering
  from an error later in lowering.
- Before modifying quantizer annotations or configuration, read
  `examples/arm/quantizer_tutorial.ipynb` for the quantization workflow and
  extension examples. Keep the tutorial aligned with quantizer API changes.
- Missing quantization parameters generally belong in the quantizer, not in
  downstream code that reconstructs them. Decompositions whose resulting
  operators need separate quantization parameters belong before annotation.
- Handle memory-layout issues in the existing layout and permute-propagation
  passes. Lowering retraces with fake tensors, not real tensor data; preserve
  dtype metadata and both static and symbolic shapes through rewrites.
- Keep decompositions and complex graph rewrites out of node visitors; perform
  them in earlier passes so visitors remain serialization-focused.
- Respect pass dependencies and ordering instead of repeating work already
  guaranteed by an earlier pass.
- Preserve existing behavior; notify the user of intentional compatibility
  changes. If a refactor would materially simplify the implementation, suggest
  it before expanding the patch.

## Static and Dynamic Shapes

When changing shape-dependent operators, symbolic arithmetic, inferred reshape
dimensions, or dynamic padding, read `backends/arm/SYMINT_ET_LOWERING.md` first.
Static dimensions are integer constants; dynamic dimensions are represented by
`SymInt` expressions and FX producers such as `aten.sym_size.int`. TOSA consumes
one-dimensional shape operands, not scalar symbolic values: dynamic dimensions
lower to `DIM`, constants to `CONST_SHAPE`, and mixed shape lists to
`CONCAT_SHAPE` when materialized.

Symbolic representation alone does not imply dynamic-shape support for every
target or operator. Check the selected TOSA specification, extensions, and
partitioner restrictions before accepting a dynamic-shape case.

Keep responsibilities separate: `ResolveViewCopyInferredDimPass` resolves `-1`
and materializes inferred symbolic dimensions into FX producers before
`SymbolicToTosaShapesPass` lowers them to TOSA shape operands.
`SymbolMaterializationHelpers` builds and reuses TOSA shape nodes; it does not
lower arbitrary raw `SymInt` values. Preserve shape metadata and test both
static and dynamic cases. Exercise the same dynamically exported program with
multiple valid input sizes, including sizes different from the export example,
to detect accidental specialization. Keep sizes within the exported constraints.

## Community-Facing Examples

Examples in `examples/arm/` are a front door for the community to get started
with and evaluate ExecuTorch, not just internal test fixtures. For Arm TOSA,
VGF, and Ethos-U examples:

- Make the documented path easy and intuitive to follow without internal
  knowledge. State prerequisites, target requirements, commands, expected
  artifacts, and how to recognize successful execution.
- Keep the main export-to-execution flow readable and focused. Put optional
  tuning and advanced configurations after the working baseline, rather than
  obscuring it with internal abstractions or test-only machinery.
- Keep scripts, notebooks, README commands, and backend APIs consistent when
  changing interfaces. Start with `examples/arm/README.md` and the affected
  example's own README.
- Maintain automated coverage of the documented path so examples do not go
  stale. Use lightweight smoke/export tests where possible and target execution
  tests where dependencies are available; check meaningful outputs, not just
  successful imports. Reuse existing example tests such as
  `backends/arm/test/misc/test_silero_vad_example.py` where applicable.
- Keep hardware/toolchain-dependent coverage explicit and report untested
  steps. Mocked helper tests do not establish that the full example works.

`examples/arm/setup.sh` installs dependencies such as compilers, Fixed Virtual
Platforms (FVPs), and drivers; `examples/arm/arm-scratch/setup_path.sh` sets up
tool paths. Use the configured project environment and setup instructions.

## Choosing Tests

Use pipelines from `backends/arm/test/tester/test_pipeline.py` and follow nearby
tests rather than assembling a separate lowering flow:

| Change or target | Test entry point and dependencies |
|------------------|-----------------------------------|
| Individual graph pass | `backends/arm/test/passes/` with `PassPipeline`; e.g. `test_remove_clone_pass.py` |
| TOSA operator correctness | `backends/arm/test/ops/test_add.py` shows `TosaPipelineFP` and `TosaPipelineINT`; numerical execution uses the TOSA reference model, not an FVP |
| Ethos-U execution | `EthosU55PipelineINT` / `EthosU65PipelineINT` use Corstone-300; `EthosU85PipelineINT` uses Corstone-320; requires Vela, target runner, and the matching FVP |
| VGF execution | `VgfPipeline` compiles with Model Converter and compares outputs through the Vulkan runtime; requires the matching runner and usable Vulkan environment |

`run_on_tosa_ref_model=False`, `run_on_fvp=False`, and
`run_on_vulkan_runtime=False` omit their respective execution checks. Do not
describe those runs as numerical validation on that reference model or target.
Preserve existing dependency markers and expected-failure conventions.

For example, after activating the project environment and Arm tool paths:

```bash
pytest -c backends/arm/test/pytest.ini backends/arm/test/ops/test_add.py -k tosa
pytest -c backends/arm/test/pytest.ini backends/arm/test/passes/test_remove_clone_pass.py
```

Start with the test relevant to the change, then broaden to affected targets,
specifications, and shape modes as needed.

## Regression Checks

Consider numerical accuracy, final-model performance, and AoT lowering time for
every behavior change. Nontrivial decomposition changes need accuracy and
performance evaluation. For graph-wide quadratic algorithms, evaluate whether a
more efficient approach is practical. Suggest measurements and alternatives
when a regression is plausible; do not claim unmeasured results.

Before finalizing non-prototyping implementation changes, run the configured
pre-push hook (`.git/hooks/pre-push`, or the hook under `core.hooksPath` in a
worktree) through the project environment and address failures caused by the
change. If execution requires approval or is blocked, report that rather than
claiming it passed.
For documentation-only changes, use focused documentation or skill validation
and whitespace checks instead; explanations without file changes need no hook.
