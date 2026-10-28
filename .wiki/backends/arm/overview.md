---
title: Arm Backend Reference Map
category: BACKEND_CONSTRAINT
backends: [Arm]
last_validated: 2026-10-23
source_issues: []
---

# Arm Backend Reference Map

This wiki keeps debugging lessons, not a second backend manual. Architecture,
setup commands, supported operators, and example code belong in the maintained
references below. Validation here means comparison with this checkout's code
and documentation, not execution on every target.

## Current Entry Points

| Topic | Reference |
|-------|-----------|
| Development decisions, partitioning, layout, and test selection | [Arm skill](../../../.claude/skills/arm/SKILL.md) |
| Ethos-U setup and first deployment | [Getting started](../../../docs/source/backends/arm-ethos-u/tutorials/ethos-u-getting-started.md) |
| Ethos-U configuration and runtime memory | [Backend overview](../../../docs/source/backends/arm-ethos-u/arm-ethos-u-overview.md) |
| VGF setup and deployment | [Getting started](../../../docs/source/backends/arm-vgf/tutorials/vgf-getting-started.md) and [backend overview](../../../docs/source/backends/arm-vgf/arm-vgf-overview.md) |
| Quantizer customization | [Quantizer tutorial](../../../examples/arm/quantizer_tutorial.ipynb) |
| Static and symbolic shapes | [Symbolic TOSA shape lowering](../../../backends/arm/SYMINT_ET_LOWERING.md) |
| Runnable community examples | [Arm examples](../../../examples/arm/README.md) |
| Cortex-M / CMSIS-NN development | [Cortex-M skill](../../../.claude/skills/cortex-m/SKILL.md) |

## Distinctions That Matter When Debugging

- **TOSA is an intermediate representation.** Ethos-U compiles it with Vela;
  VGF uses Model Converter. A successful TOSA reference-model comparison does
  not establish successful compilation or execution on either target.
- **Cortex-M is not another output mode of the Ethos-U partitioner.** Its
  CMSIS-NN backend has its own quantizer and graph transformations. Nodes left
  outside an Ethos-U delegate still require appropriate runtime kernels.
- **Shape support is target-specific.** TOSA symbolic-shape lowering exists;
  the Ethos-U partitioner currently requires resolved tensor shapes, while VGF
  does not impose that same check. The selected specification, extensions, and
  operator checks still determine whether a particular graph is accepted.
- **Support is configuration-specific.** Use current partitioner checks and
  generated support documents, then reproduce the specific case.
  Tests describe covered configurations, not every possible model.

## Operator Support References

- [Ethos-U55 operator support](../../../docs/source/backends/arm-ethos-u/U55_op_support.md)
- [Ethos-U85 operator support](../../../docs/source/backends/arm-ethos-u/U85_op_support.md)
- [VGF operator support](../../../docs/source/backends/arm-vgf/VGF_op_support.md)
- [Support-check implementation](../../../backends/arm/operator_support/tosa_supported_operators.py)

For a different release, consult that release's documentation and dependency
pins rather than mixing current instructions with its source tree.
