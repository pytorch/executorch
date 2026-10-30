---
title: Arm Backend Debugging Rules
category: BACKEND_CONSTRAINT
backends: [Arm]
last_validated: 2026-10-23
source_issues: []
---

# Arm Backend — Debugging Guardrails

These guardrails were checked against this checkout, not all released versions.
Use the [reference map](../backends/arm/overview.md) for current documentation.
For implementation work, use the [Arm skill](../../.claude/skills/arm/SKILL.md).

1. **Identify the target first.** TOSA reference execution, VGF execution,
   Ethos-U execution, and Cortex-M/CMSIS-NN are distinct paths. Passing one does
   not validate the others.
2. **Shape support is conditional.** Ethos-U currently requires resolved tensor
   shapes; TOSA/VGF shape support depends on specification and operator checks.
3. **Explain non-delegation before changing support.** Inspect rejection reasons
   and delegation tags. Accepted nodes must actually lower, and fallback nodes
   still need compatible runtime kernels.
4. **Use current support evidence.** Consult generated operator documents,
   support checks, and tests for the selected configuration.
   Coverage of one configuration is not universal support.
5. **Preserve quantization and layout semantics.** Inspect annotation and Q/DQ
   metadata for accuracy failures. Layout permutes are introduced around
   channels-last nodes and can cancel during propagation; remaining conversions
   are not automatically bugs.
6. **Separate simulator, hardware, and compiler results.** Check I/O dtype,
   memory configuration, and the deployed runner when hardware differs from an
   FVP. Compiler estimates are not measured execution performance.
7. **Match setup to the checkout.** Use current requirements, optional extras,
   and setup documentation. Resolve package conflicts rather than assuming
   incompatible dependency versions will work together.
8. **Verify input contracts.** Match the program's input dtype, shape, and
   quantization parameters. An int8-I/O program needs correctly quantized input
   values, not the floating-point preprocessing used for a different program.
