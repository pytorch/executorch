#!/usr/bin/env python3
# Copyright 2026 Meta Platforms, Inc. and affiliates.
# pyre-strict
"""
Comparison test for FuseTosaTransposesPass optimization.

This script compares TRANSPOSE counts before and after FuseTosaTransposesPass
to verify the optimization is effective.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from executorch.backends.arm._passes import (
    AnnotateOutputDimOrderPass,
    FuseTosaTransposesPass,
    ToTosaMemoryFormatPass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass


def count_transposes(graph_module: torch.fx.GraphModule) -> int:
    """Count the number of TOSA TRANSPOSE operations in a graph."""
    count = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and "TRANSPOSE" in str(node.target):
            count += 1
    return count


class ConvReluConvChain(nn.Module):
    """
    A chain of Conv2D + ReLU layers that generates many TRANSPOSE ops.
    This pattern mimics the EMG CC model TDS blocks.
    """

    def __init__(self, num_blocks: int = 6) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.rand(1, 16, 8, 8),)


class LargeModel(nn.Module):
    """
    A larger model with multiple convolutions, pooling, and FC layers.
    Designed to generate many intermediate TRANSPOSE operations.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)

        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))

        return x

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.rand(1, 16, 32, 32),)


def run_comparison(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    model_name: str
) -> Dict[str, int]:
    """
    Run comparison of TRANSPOSE counts with and without FuseTosaTransposesPass.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Run pipeline WITHOUT FuseTosaTransposesPass (baseline)
    print("\n[1] Running WITHOUT FuseTosaTransposesPass...")
    pipeline_baseline = PassPipeline(
        model,
        inputs,
        pass_list=[RemoveGetItemPass, AnnotateOutputDimOrderPass],
        passes_with_exported_program=[ToTosaMemoryFormatPass],
    )
    pipeline_baseline.pop_stage("run_method_and_compare_outputs")
    result_baseline = pipeline_baseline.run()

    baseline_count = count_transposes(result_baseline.graph_module)
    print(f"    TRANSPOSE ops (baseline): {baseline_count}")

    # Run pipeline WITH FuseTosaTransposesPass (optimized)
    print("\n[2] Running WITH FuseTosaTransposesPass...")
    pipeline_optimized = PassPipeline(
        model,
        inputs,
        pass_list=[RemoveGetItemPass, AnnotateOutputDimOrderPass],
        passes_with_exported_program=[
            ToTosaMemoryFormatPass,
            FuseTosaTransposesPass,
        ],
    )
    pipeline_optimized.pop_stage("run_method_and_compare_outputs")
    result_optimized = pipeline_optimized.run()

    optimized_count = count_transposes(result_optimized.graph_module)
    print(f"    TRANSPOSE ops (optimized): {optimized_count}")

    # Calculate reduction
    reduction = baseline_count - optimized_count
    reduction_pct = (reduction / baseline_count * 100) if baseline_count > 0 else 0

    print(f"\n[3] Results Summary:")
    print(f"    Baseline:    {baseline_count} TRANSPOSE ops")
    print(f"    Optimized:   {optimized_count} TRANSPOSE ops")
    print(f"    Reduction:   {reduction} ops ({reduction_pct:.1f}%)")

    return {
        "model": model_name,
        "baseline": baseline_count,
        "optimized": optimized_count,
        "reduction": reduction,
        "reduction_pct": reduction_pct,
    }


def main() -> None:
    """Run TRANSPOSE count comparison tests."""
    torch.manual_seed(42)

    print("\n" + "="*60)
    print(" FuseTosaTransposesPass Optimization Comparison")
    print("="*60)

    results = []

    # Test 1: Conv-ReLU-Conv chain (6 blocks)
    model1 = ConvReluConvChain(num_blocks=6)
    model1.eval()
    results.append(run_comparison(model1, model1.get_inputs(), "ConvReluConvChain (6 blocks)"))

    # Test 2: Conv-ReLU-Conv chain (12 blocks)
    model2 = ConvReluConvChain(num_blocks=12)
    model2.eval()
    results.append(run_comparison(model2, model2.get_inputs(), "ConvReluConvChain (12 blocks)"))

    # Test 3: Large model with multiple conv layers
    model3 = LargeModel()
    model3.eval()
    results.append(run_comparison(model3, model3.get_inputs(), "LargeModel"))

    # Print summary table
    print("\n" + "="*60)
    print(" SUMMARY TABLE")
    print("="*60)
    print(f"{'Model':<35} {'Baseline':>10} {'Optimized':>10} {'Reduction':>12}")
    print("-"*60)

    total_baseline = 0
    total_optimized = 0

    for r in results:
        print(f"{r['model']:<35} {r['baseline']:>10} {r['optimized']:>10} {r['reduction']:>8} ({r['reduction_pct']:.1f}%)")
        total_baseline += r["baseline"]
        total_optimized += r["optimized"]

    total_reduction = total_baseline - total_optimized
    total_pct = (total_reduction / total_baseline * 100) if total_baseline > 0 else 0

    print("-"*60)
    print(f"{'TOTAL':<35} {total_baseline:>10} {total_optimized:>10} {total_reduction:>8} ({total_pct:.1f}%)")
    print("="*60)

    # Estimate cycle reduction
    # Based on plan: ~10% of NPU cycles from transposes, near-zero after optimization
    estimated_cycle_reduction = total_pct * 0.1  # ~10% of cycles from transposes
    print(f"\nEstimated NPU cycle reduction: ~{estimated_cycle_reduction:.1f}%")
    print("(Based on ~10% of NPU cycles spent on TRANSPOSE operations)")


if __name__ == "__main__":
    main()
