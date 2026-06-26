# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU compile specification for ExecuTorch."""

from __future__ import annotations

from executorch.exir.backend.compile_spec_schema import CompileSpec

# AXON hardware constraints (from nrf_axon_nn_compiler_types.h and Nordic docs)
AXON_MAX_FC_INPUT = 2048
AXON_MAX_FC_OUTPUT = 2048
AXON_MAX_CONV2D_FILTER = 16      # Max filter height/width for Conv2D
AXON_MAX_CONV_STRIDE = 31
AXON_MAX_POOL_FILTER = 32        # Max filter height/width for pooling
AXON_MAX_TENSOR_DIM = 1024       # Max height/width/channels
AXON_MAX_INPUTS_PER_NODE = 2


class AxonCompileSpec:
    """Configuration for compiling models targeting the AXON NPU.

    Args:
        sdk_edge_ai_path: Path to Nordic sdk-edge-ai repo. Can also be
            set via the ``SDK_EDGE_AI_PATH`` environment variable. Required
            for compilation to AXON command buffers; not needed for TOSA
            lowering only.
        model_name: Human-readable prefix for delegated subgraphs.
            The actual C symbols in the generated headers append a
            content-derived hash suffix so multiple subgraphs in the
            same firmware build never collide.
        tosa_spec: TOSA version string (default: "TOSA-1.0+INT").
        axon_generated_dir: Where ``preprocess()`` writes the per-subgraph
            ``axon_subgraph_*.h`` files and the master
            ``axon_subgraphs_table.h``. Required when writing generated
            headers for firmware integration.
    """

    def __init__(
        self,
        sdk_edge_ai_path: str | None = None,
        model_name: str = "axon_model",
        tosa_spec: str = "TOSA-1.0+INT",
        axon_generated_dir: str | None = None,
    ):
        self.sdk_edge_ai_path = sdk_edge_ai_path
        self.model_name = model_name
        self.tosa_spec = tosa_spec
        self.axon_generated_dir = axon_generated_dir

    def to_compile_specs(self) -> list[CompileSpec]:
        """Convert to ExecuTorch CompileSpec list."""
        specs = [
            CompileSpec("tosa_spec", self.tosa_spec.encode()),
            CompileSpec("output_format", b"tosa"),
            CompileSpec("model_name", self.model_name.encode()),
        ]
        if self.sdk_edge_ai_path:
            specs.append(CompileSpec("sdk_edge_ai_path", self.sdk_edge_ai_path.encode()))
        if self.axon_generated_dir:
            specs.append(
                CompileSpec("axon_generated_dir", self.axon_generated_dir.encode())
            )
        return specs
