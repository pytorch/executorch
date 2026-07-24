# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AXON NPU backend for ExecuTorch.

Compiles delegated subgraphs for execution on Nordic's AXON NPU.
Composes with TOSABackend for TOSA lowering, then compiles TOSA to
AXON command buffers via our converter + Nordic's compiler lib.

Pipeline::

    ExportedProgram -> TOSABackend._preprocess() -> TOSA flatbuffer
        -> tosa_reader -> axon_compiler -> axon_binary -> Nordic compiler lib
        -> compiled AXON model (.h with command buffers)
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import final, List

from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from .codegen import (
    derive_subgraph_name,
    make_marker,
    regenerate_table,
    rewrite_header_symbols,
    rewrite_op_extension_symbols,
    write_subgraph_header,
)
from executorch.backends.nordic.axon_binary import AxonBinaryBuilder
from executorch.backends.nordic.axon_types import ActivationQuantInfo
from executorch.backends.nordic.axon_compiler import tosa_to_axon_layers
from executorch.backends.nordic.tosa_reader import parse_tosa_flatbuffer

logger = logging.getLogger(__name__)


def _extract_activation_quant_info(edge_program) -> list[ActivationQuantInfo]:
    """Scan the edge program FX graph for sigmoid/tanh/softmax ops and
    extract their input/output quantization parameters.

    AXON op extensions (sigmoid=101, tanh=102, softmax=100) require the
    preceding layer to output INT16 q3.12 (sigmoid/tanh) or INT32 q11.12
    (softmax). To recompute the rescale, we need the input/output scales
    of the activation, which are NOT directly available in the TOSA graph
    (TOSA uses TABLE ops with the scales baked in). So we extract them
    here from the still-quantize/dequantize-annotated FX graph, before
    TOSA lowering destroys that information.

    Returns:
        List of ActivationQuantInfo records, in the order activations
        appear in the graph.
    """
    info: list[ActivationQuantInfo] = []
    nodes = list(edge_program.graph_module.graph.nodes)

    sigmoid_targets = ("aten.sigmoid", "aten_sigmoid")
    tanh_targets = ("aten.tanh", "aten_tanh")
    softmax_targets = ("aten.softmax", "aten._softmax", "aten_softmax", "aten__softmax")
    amax_targets = ("aten.amax", "aten_amax")

    def matches(node, names):
        s = str(node.target)
        return any(n in s for n in names)

    def get_quant_args(qnode):
        if qnode is None or len(qnode.args) < 3:
            return None
        scale = qnode.args[1]
        zp = qnode.args[2]
        return (float(scale), int(zp))

    def find_input_quant(act_node):
        if not act_node.args:
            return None
        cur = act_node.args[0]
        for _ in range(4):
            if cur is None or not hasattr(cur, "op"):
                return None
            if "dequantize" in str(cur.target):
                cur = cur.args[0] if cur.args else None
                continue
            if "quantize" in str(cur.target):
                return get_quant_args(cur)
            return None
        return None

    def find_output_quant(act_node):
        for user in act_node.users:
            if "quantize" in str(user.target) and "dequantize" not in str(user.target):
                return get_quant_args(user)
        return None

    def find_softmax_output_quant(amax_node):
        seen = set()
        frontier = [amax_node]
        steps = 0
        while frontier and steps < 30:
            steps += 1
            nxt = []
            for n in frontier:
                for u in n.users:
                    if id(u) in seen:
                        continue
                    seen.add(id(u))
                    s = str(u.target)
                    if "aten" in s and "mul" in s and "tensor" in s.lower():
                        return find_output_quant(u)
                    nxt.append(u)
            frontier = nxt
        return None

    for node in nodes:
        if node.op != "call_function":
            continue
        if matches(node, sigmoid_targets):
            op_type = "sigmoid"
        elif matches(node, tanh_targets):
            op_type = "tanh"
        elif matches(node, softmax_targets):
            op_type = "softmax"
        elif matches(node, amax_targets):
            in_q = find_input_quant(node)
            out_q = find_softmax_output_quant(node)
            if in_q is None or out_q is None:
                continue
            info.append(ActivationQuantInfo(
                op_type="softmax",
                input_scale=in_q[0], input_zp=in_q[1],
                output_scale=out_q[0], output_zp=out_q[1],
            ))
            logger.debug(f"  Activation softmax (from amax): in_scale={in_q[0]:.6f} "
                         f"in_zp={in_q[1]} out_scale={out_q[0]:.6f} out_zp={out_q[1]}")
            continue
        else:
            continue

        in_q = find_input_quant(node)
        out_q = find_output_quant(node)
        if in_q is None or out_q is None:
            logger.warning(f"  Could not extract quant info for {op_type} ({node.name})")
            continue

        info.append(ActivationQuantInfo(
            op_type=op_type,
            input_scale=in_q[0],
            input_zp=in_q[1],
            output_scale=out_q[0],
            output_zp=out_q[1],
        ))
        logger.debug(f"  Activation {op_type}: in_scale={in_q[0]:.6f} in_zp={in_q[1]} "
                     f"out_scale={out_q[0]:.6f} out_zp={out_q[1]}")

    return info


@final
class AxonBackend(BackendDetails):
    """ExecuTorch backend for Nordic AXON NPU.

    Follows the same composition pattern as EthosUBackend:
    reuses TOSABackend for TOSA lowering, then compiles
    TOSA -> AXON command buffers.
    """

    @staticmethod
    def preprocess(
        edge_program,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Compile a delegated subgraph for AXON NPU execution.

        Args:
            edge_program: ExportedProgram from ExecuTorch edge lowering.
            compile_specs: List of CompileSpec. We look for:
                - tosa_spec: TOSA version/profile (default: TOSA-1.0+INT)
                - sdk_edge_ai_path: Path to Nordic sdk-edge-ai repo
                - model_name: Name for the compiled model
                - axon_generated_dir: Where to write generated headers

        Returns:
            PreprocessResult with AXON marker as processed_bytes.
        """
        # Parse compile specs
        sdk_path = os.environ.get("SDK_EDGE_AI_PATH", "")
        model_name = "axon_model"
        tosa_spec_str = "TOSA-1.0+INT"
        generated_dir_override: str | None = None

        for spec in compile_specs:
            if spec.key == "sdk_edge_ai_path":
                sdk_path = spec.value.decode() if isinstance(spec.value, bytes) else spec.value
            elif spec.key == "model_name":
                model_name = spec.value.decode() if isinstance(spec.value, bytes) else spec.value
            elif spec.key == "tosa_spec":
                tosa_spec_str = spec.value.decode() if isinstance(spec.value, bytes) else spec.value
            elif spec.key == "axon_generated_dir":
                generated_dir_override = (
                    spec.value.decode() if isinstance(spec.value, bytes) else spec.value
                )

        logger.info("AxonBackend.preprocess: model=%s, tosa_spec=%s",
                    model_name, tosa_spec_str)

        try:
            return AxonBackend._do_preprocess(
                edge_program, model_name, tosa_spec_str,
                sdk_path, generated_dir_override,
            )
        except Exception:
            logger.exception("AxonBackend.preprocess failed for model=%s", model_name)
            # Return marker-only so the .pte is still valid (firmware will
            # report "subgraph not found" instead of crashing at load time).
            return PreprocessResult(
                processed_bytes=make_marker(
                    derive_subgraph_name(model_name, b"error")
                )
            )

    @staticmethod
    def _do_preprocess(
        edge_program,
        model_name: str,
        tosa_spec_str: str,
        sdk_path: str,
        generated_dir_override: str | None,
    ) -> PreprocessResult:
        """Internal preprocess implementation. Separated so the public
        ``preprocess()`` can catch exceptions and return a safe fallback."""
        # Extract activation quantization info BEFORE TOSA lowering.
        activation_info = _extract_activation_quant_info(edge_program)
        if activation_info:
            logger.info("Found %d activation op(s) for AXON op extensions",
                       len(activation_info))

        # 1. Reuse TOSA lowering (shared with Ethos-U, VGF, etc.)
        tosa_spec = TosaSpecification.create_from_string(tosa_spec_str)
        tosa_compile_spec = TosaCompileSpec(tosa_spec)
        tosa_result = TOSABackend._preprocess(edge_program, tosa_compile_spec)
        tosa_flatbuffer = tosa_result.processed_bytes
        logger.info("TOSA lowering produced %d bytes", len(tosa_flatbuffer))

        # Save TOSA flatbuffer for debugging (unique per model_name)
        tosa_debug_path = os.path.join(
            tempfile.gettempdir(),
            f"axon_tosa_debug_{model_name}.tosa",
        )
        with open(tosa_debug_path, "wb") as f:
            f.write(tosa_flatbuffer)

        # 2. Parse TOSA -> AXON layers
        graph = parse_tosa_flatbuffer(tosa_flatbuffer)
        layers = tosa_to_axon_layers(graph, activation_info=activation_info)
        logger.info("Converted to %d AXON layers", len(layers))

        # 3. Pack intermediate binary, derive stable subgraph name
        builder = AxonBinaryBuilder()
        intermediate_binary = builder.build(layers, model_name=model_name)
        logger.info("Intermediate binary: %d bytes", len(intermediate_binary))

        subgraph_name = derive_subgraph_name(model_name, intermediate_binary)
        logger.info("Subgraph unique name: %s", subgraph_name)

        # 4. Call Nordic compiler lib
        import platform
        system = platform.system()
        lib_names = {
            "Linux": "libnrf-axon-nn-compiler-lib-amd64.so",
            "Darwin": "libnrf-axon-nn-compiler-lib-arm64.dylib",
            "Windows": "nrf-axon-nn-compiler-lib-amd64.dll",
        }
        lib_name = lib_names.get(system, lib_names["Linux"])
        compiler_lib_path = os.path.join(
            sdk_path, "tools", "axon", "compiler", "bin", system, lib_name
        )

        if not os.path.exists(compiler_lib_path):
            logger.warning("AXON compiler lib not found: %s", compiler_lib_path)
            logger.warning(
                "Returning marker only — firmware will not find a model. "
                "Set SDK_EDGE_AI_PATH to your Nordic sdk-edge-ai directory."
            )
            return PreprocessResult(processed_bytes=make_marker(subgraph_name))

        # Re-pack with the unique name baked into the binary
        intermediate_binary = builder.build(layers, model_name=subgraph_name)

        # Write intermediate binary to temp file, compile, read result
        compile_dir = tempfile.mkdtemp(prefix="axon_compile_")
        bin_path = os.path.join(compile_dir, f"{subgraph_name}.bin")
        with open(bin_path, "wb") as f:
            f.write(intermediate_binary)

        output_prefix = os.path.join(compile_dir, f"nrf_axon_model_{subgraph_name}")

        from executorch.backends.nordic.axon_compiler import _call_compiler_lib
        result = _call_compiler_lib(compiler_lib_path, bin_path, output_prefix)

        if result != 0:
            logger.error("AXON compilation failed with code %d", result)
            return PreprocessResult(processed_bytes=make_marker(subgraph_name))

        # Read the compiled header file
        header_path = f"{output_prefix}_.h"
        if not os.path.exists(header_path):
            logger.error("Compiled header not found: %s", header_path)
            return PreprocessResult(processed_bytes=make_marker(subgraph_name))

        with open(header_path, "rb") as f:
            compiled_header = f.read().decode("utf-8", errors="replace")
        logger.info("AXON compilation successful: %d bytes header",
                    len(compiled_header))

        # Rewrite op extension symbols to generic names
        compiled_header = rewrite_op_extension_symbols(compiled_header)

        # Rewrite header symbols for unique naming
        compiled_header = rewrite_header_symbols(
            compiled_header, subgraph_name, subgraph_name,
        )

        # 5. Write per-subgraph .h and regenerate master table
        if generated_dir_override:
            from pathlib import Path
            generated_dir = Path(generated_dir_override)
            write_subgraph_header(generated_dir, subgraph_name, compiled_header)
            regenerate_table(generated_dir)
        else:
            logger.info(
                "No axon_generated_dir specified — skipping header generation. "
                "Set axon_generated_dir in AxonCompileSpec to write firmware headers."
            )

        # 6. Return the marker as processed_bytes
        return PreprocessResult(processed_bytes=make_marker(subgraph_name))
