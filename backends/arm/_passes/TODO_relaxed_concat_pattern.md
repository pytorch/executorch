# TODO: Investigate relaxed transpose-concat pattern matching (multiple users)

## Background

`PropagateTransposesThroughConcatPass` was added (commit `e7b5bd6d`) to target the pattern:
```
[T(perm), T(perm), ...] → Concat(dim=d) → T(inv_perm)
```

Current pass requirements:
1. All Concat inputs must be transposes with the **same** permutation
2. All input transposes must have **only one user** (the Concat)
3. Output transpose must have the **inverse** permutation

## Problem

In the Control Ceres model, the actual patterns have:
- Input transposes with **multiple users** (not just Concat)
- Input transposes with **different permutations**

This prevents the pass from intercepting patterns in Control Ceres.

Example from Vela command stream:
```
766   Transpose            tosa_transpose_default_5      
767   Transpose            tosa_transpose_default        
768   Concat               aten_cat_default              
769   Transpose            tosa_transpose_default_6
```

## Investigation Scope

1. **Analyze the actual graph patterns** in Control Ceres to understand:
   - What users do the input transposes have besides Concat?
   - What are the permutations of the input transposes?

2. **Evaluate relaxation options**:
   - **Option A**: Allow input transposes with multiple users
     - Requires duplicating the transpose for other users
     - May increase graph size but reduce total transpose ops
   - **Option B**: Handle mixed permutations by only propagating matching subsets
     - Only some inputs participate in the optimization
   - **Option C**: Target a different pattern entirely
     - E.g., single-input concat optimization

3. **Assess impact**:
   - Correctness: Ensure no semantic changes
   - Performance: Measure actual cycle reduction on Ethos-U55

## Related Work

- `FuseTransposeSandwichPass`: Targets T1 → op → T2 patterns
- `PropagateTransposesThroughRescalePass`: Targets T1 → Rescale → T2 patterns
- `FuseConsecutiveTransposesPass`: Targets T1 → T2 sequences

## Files

- `/fbcode/executorch/backends/arm/_passes/propagate_transposes_through_concat_pass.py`
- `/fbcode/executorch/backends/arm/_passes/arm_pass_manager.py`

## Test Command

```bash
buck2 test @fbcode//mode/dev fbcode//frl/ctrl/torchstream/torchstream/pt2/tests:test_pt2_emg_lowering -- 'test_combined_control_ceres_u55' --print-passing-details
```

## BEFORE Metrics (baseline)

- NPU cycles: 1,356,652 cycles/batch
- Total cycles: 1,369,354 cycles/batch
- Total SRAM used: 2,103.83 KiB
- NPU operators: 367 (100.0%)
- Total Transpose ops: 536 (tosa_transpose: 273, aten_permute_copy: 278)

## Author

eliamesefe@meta.com

## Date Created

2026-03-05
