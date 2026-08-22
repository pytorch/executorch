# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal standalone Arm/VGF custom-shader examples.

This example shows the full stack for two GLSL operators:
- a scalar buffer-backed operator
- an RGBA image-backed operator
- the PyTorch fake implementation needed for export
- the `tosa.CUSTOM` fake implementation needed for lowering
- a small rewrite pass that wraps the custom op as `tosa.CUSTOM`
- VGF lowering and runtime execution against the produced `.pte`

Prerequisites:
- `glslc` available on `PATH`
- the Arm `model_converter` tools installed and available to the VGF backend
- a runtime build exposing `VgfBackend`
"""

from __future__ import annotations

import base64
import json
import operator
import shutil
import subprocess  # nosec B404 - fixed local tool invocation
from pathlib import Path
from typing import Callable, cast

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes import ArmPass, RewriteMatmulPass
from executorch.backends.arm._passes.arm_pass_manager import (
    register_pass_insertions_after,
)
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.dialect.ops.custom import (
    has_fake_tosa_impl,
    register_fake_tosa,
)
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.export_util.utils import save_pte_program
from executorch.runtime import Runtime
from torch.export import export
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.library import register_fake

CUSTOM_NAMESPACE = "arm_example_custom_shader"
SCALE_ADD_OPERATOR = f"{CUSTOM_NAMESPACE}::scale_add"
RGBA_BIAS_OPERATOR = f"{CUSTOM_NAMESPACE}::rgba_bias"
TOSA_SCALE_ADD_OPERATOR = "examples.arm.scale_add"
TOSA_RGBA_BIAS_OPERATOR = "examples.arm.rgba_bias"
CUSTOM_DOMAIN = "com.arm.VulkanCustomShader"
ARTIFACT_DIR = Path("arm_custom_operator_vgf")
SCALE_ADD_PTE_NAME = "scale_add_vgf.pte"
RGBA_BIAS_PTE_NAME = "rgba_bias_vgf.pte"

TensorUnary = Callable[[torch.Tensor], torch.Tensor]

_SCALE_ADD_SHADER_SOURCE_NAME = "scale_add.comp"
_SCALE_ADD_SPIRV_NAME = "scale_add.spv"
_SCALE_ADD_SHADER_SOURCE = """#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer In { float x[]; };
layout(set = 0, binding = 1) buffer Out { float y[]; };
void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= y.length()) {
    return;
  }
  y[idx] = x[idx] * 2.0 + 5.0;
}
"""

_RGBA_BIAS_SHADER_SOURCE_NAME = "rgba_bias.comp"
_RGBA_BIAS_SPIRV_NAME = "rgba_bias.spv"
_RGBA_BIAS_SHADER_SOURCE = """#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D in_image;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D out_image;
void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(out_image);
  if (coord.x >= size.x || coord.y >= size.y) {
    return;
  }
  vec2 uv = (vec2(coord) + vec2(0.5)) / vec2(size);
  vec4 value = texture(in_image, uv);
  imageStore(out_image, coord, value + vec4(10.0, 20.0, 30.0, 40.0));
}
"""


def _build_scale_add_payload(output_dir: Path) -> list[int]:
    payload = {
        "entry_point": "main",
        "workgroup_sizes": [64, 1, 1],
        "is_vkshader": True,
        "shader_code": _compile_shader(
            output_dir,
            _SCALE_ADD_SHADER_SOURCE_NAME,
            _SCALE_ADD_SPIRV_NAME,
            _SCALE_ADD_SHADER_SOURCE,
        ),
        "shader_language": "SPIR-V",
        "push_constants": "",
        "input_0_binding": 0,
        "output_0_binding": 1,
        "input_0_type": "Buffer",
        "output_0_type": "Buffer",
        "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        "input_0_descriptorset": 0,
        "output_0_descriptorset": 0,
        "input_0_vkformat": "VK_FORMAT_R32_SFLOAT",
        "output_0_vkformat": "VK_FORMAT_R32_SFLOAT",
    }
    return list(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _build_rgba_bias_payload(output_dir: Path) -> list[int]:
    payload = {
        "entry_point": "main",
        "workgroup_sizes": [8, 8, 1],
        "is_vkshader": True,
        "shader_code": _compile_shader(
            output_dir,
            _RGBA_BIAS_SHADER_SOURCE_NAME,
            _RGBA_BIAS_SPIRV_NAME,
            _RGBA_BIAS_SHADER_SOURCE,
        ),
        "shader_language": "SPIR-V",
        "push_constants": "",
        "input_0_binding": 0,
        "output_0_binding": 1,
        "input_0_type": "Image",
        "output_0_type": "Image",
        "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
        "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
        "input_0_descriptorset": 0,
        "output_0_descriptorset": 0,
        "input_0_vkformat": "VK_FORMAT_R32G32B32A32_SFLOAT",
        "output_0_vkformat": "VK_FORMAT_R32G32B32A32_SFLOAT",
        "input_0_sampler": {
            "mag_filter": "VK_FILTER_LINEAR",
            "min_filter": "VK_FILTER_LINEAR",
            "address_mode_u": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
            "address_mode_v": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
            "border_color": "VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK",
        },
    }
    return list(json.dumps(payload, sort_keys=True).encode("utf-8"))


def register_example_custom_op() -> None:
    """Register the Python-side pieces of the custom-op contract.

    The custom shader flow has two layers of operator identity:

    1. A normal `torch.library` op used by the eager model and by export.
    2. A `tosa.CUSTOM` operator name plus payload used by Arm lowering.

    Both layers need their own fake implementations:
    - the PyTorch fake keeps export/shape propagation working before rewrite
    - the TOSA fake keeps `tosa.CUSTOM` shape propagation working after rewrite
    """

    # Step 1: register the user-visible library op together with its eager
    # implementation. `@torch.library.custom_op` defines the library schema
    # directly from the Python signature, so there is no separate `.define(...)`
    # call in this example.
    @torch.library.custom_op(SCALE_ADD_OPERATOR, mutates_args=())
    def _scale_add_impl(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 5.0

    # Step 2: register the PyTorch fake for the library op. Export uses this
    # for metadata propagation before we rewrite the op to `tosa.CUSTOM`.
    def _scale_add_fake_impl(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    cast(TensorUnary, register_fake(SCALE_ADD_OPERATOR)(_scale_add_fake_impl))

    # Step 3: register the stable TOSA custom operator name used by the Arm
    # lowering. This name must match the `operator_name` that the rewrite pass
    # emits into the `tosa.CUSTOM` node.
    #
    # The TOSA dialect schema is:
    #   CUSTOM(Tensor[] inputs, str operator_name, str domain_name,
    #          int[] implementation_attrs) -> Tensor[]
    #
    # The dialect helper unwraps the outer `Tensor[]` before invoking the fake,
    # so this fake receives `inputs=[x]`, not `inputs=[[x]]`. The fake must
    # still return a list because `tosa.CUSTOM` is a list-valued op.
    @register_fake_tosa(TOSA_SCALE_ADD_OPERATOR)
    def _scale_add_tosa_fake(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        assert operator_name == TOSA_SCALE_ADD_OPERATOR
        assert domain_name == CUSTOM_DOMAIN
        _ = implementation_attrs
        return [torch.empty_like(inputs[0])]

    # Steps 4-6: register a second library op that uses RGBA storage images
    # internally. The eager op still uses the normal graph-visible NCHW shape;
    # the rewrite pass adds the NCHW <-> NHWC bridge around the image shader.
    @torch.library.custom_op(RGBA_BIAS_OPERATOR, mutates_args=())
    def _rgba_bias_impl(x: torch.Tensor) -> torch.Tensor:
        bias = x.new_tensor([10.0, 20.0, 30.0, 40.0]).view(1, 4, 1, 1)
        return x + bias

    def _rgba_bias_fake_impl(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    cast(TensorUnary, register_fake(RGBA_BIAS_OPERATOR)(_rgba_bias_fake_impl))

    @register_fake_tosa(TOSA_RGBA_BIAS_OPERATOR)
    def _rgba_bias_tosa_fake(
        inputs: list[torch.Tensor],
        operator_name: str,
        domain_name: str,
        implementation_attrs: list[int],
    ) -> list[torch.Tensor]:
        assert operator_name == TOSA_RGBA_BIAS_OPERATOR
        assert domain_name == CUSTOM_DOMAIN
        _ = implementation_attrs
        return [torch.empty_like(inputs[0])]


class EncodeScaleAddToTosaCustomPass(ArmPass):
    """Rewrite the library op to a `tosa.CUSTOM` node with shader payload.

    This pass is the bridge between the user-visible library op and the Arm
    custom-shader lowering contract. After partitioning has kept the library op
    inside the delegated region, this pass replaces it with:
    - a `tosa.CUSTOM` node carrying the Vulkan shader payload
    - a `getitem` extracting the single tensor output
    """

    _passes_required_after = set()

    def __init__(self, output_dir: Path) -> None:
        self._implementation_attrs = _build_scale_add_payload(output_dir)

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function" or SCALE_ADD_OPERATOR not in str(node.target):
                continue
            if not has_fake_tosa_impl(TOSA_SCALE_ADD_OPERATOR):
                raise RuntimeError(
                    f"tosa.CUSTOM fake impl is not registered for {TOSA_SCALE_ADD_OPERATOR}"
                )

            (x,) = node.args
            fake_outputs = [torch.empty_like(x.meta["val"])]
            with graph.inserting_before(node):
                custom_node = graph.call_function(
                    exir_ops.backend.tosa.CUSTOM.default,
                    args=([x],),
                    kwargs={
                        "operator_name": TOSA_SCALE_ADD_OPERATOR,
                        "domain_name": CUSTOM_DOMAIN,
                        "implementation_attrs": self._implementation_attrs,
                    },
                )
                custom_node.meta = dict(node.meta)
                _set_fake_tensor_meta(custom_node, fake_outputs)

                output = graph.call_function(
                    operator.getitem,
                    args=(custom_node, 0),
                    kwargs={},
                )
                output.meta = dict(node.meta)
                _set_fake_tensor_meta(output, fake_outputs[0])

            node.replace_all_uses_with(output)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)


class EncodeRgbaBiasToTosaCustomPass(ArmPass):
    """Rewrite the RGBA library op to `tosa.CUSTOM` with image resources."""

    _passes_required_after = set()

    def __init__(self, output_dir: Path) -> None:
        self._implementation_attrs = _build_rgba_bias_payload(output_dir)

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function" or RGBA_BIAS_OPERATOR not in str(node.target):
                continue
            if not has_fake_tosa_impl(TOSA_RGBA_BIAS_OPERATOR):
                raise RuntimeError(
                    f"tosa.CUSTOM fake impl is not registered for {TOSA_RGBA_BIAS_OPERATOR}"
                )

            (x,) = node.args
            nhwc_value = exir_ops.edge.aten.permute_copy.default(
                x.meta["val"], list(NHWC_ORDER)
            )
            fake_outputs = [torch.empty_like(nhwc_value)]
            with graph.inserting_before(node):
                nhwc_input = graph.call_function(
                    exir_ops.edge.aten.permute_copy.default,
                    args=(x, list(NHWC_ORDER)),
                    kwargs={},
                )
                nhwc_input.meta = dict(x.meta)
                _set_fake_tensor_meta(nhwc_input, nhwc_value)

                custom_node = graph.call_function(
                    exir_ops.backend.tosa.CUSTOM.default,
                    args=([nhwc_input],),
                    kwargs={
                        "operator_name": TOSA_RGBA_BIAS_OPERATOR,
                        "domain_name": CUSTOM_DOMAIN,
                        "implementation_attrs": self._implementation_attrs,
                    },
                )
                custom_node.meta = dict(node.meta)
                _set_fake_tensor_meta(custom_node, fake_outputs)

                nhwc_output = graph.call_function(
                    operator.getitem,
                    args=(custom_node, 0),
                    kwargs={},
                )
                nhwc_output.meta = dict(node.meta)
                _set_fake_tensor_meta(nhwc_output, fake_outputs[0])

                output = graph.call_function(
                    exir_ops.edge.aten.permute_copy.default,
                    args=(nhwc_output, list(NHWC_INVERSE_ORDER)),
                    kwargs={},
                )
                output.meta = dict(node.meta)
                _set_fake_tensor_meta(
                    output,
                    exir_ops.edge.aten.permute_copy.default(
                        fake_outputs[0], list(NHWC_INVERSE_ORDER)
                    ),
                )

            node.replace_all_uses_with(output)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)


class ScaleAddModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_example_custom_shader.scale_add.default(x)


class RgbaBiasModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_example_custom_shader.rgba_bias.default(x)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Steps 1-3: register a custom op into the torch.library and enable
    # ArmBackend handling.
    register_example_custom_op()

    # Install the rewrite passes once up front. Each lowering block below then
    # registers the relevant library op with its own partitioner instance.
    # `register_pass_insertions_after(...)` updates global Arm pass state, which
    # is acceptable here because this is a standalone example script.
    register_pass_insertions_after(
        RewriteMatmulPass,
        [
            EncodeScaleAddToTosaCustomPass(ARTIFACT_DIR / "scale_add"),
            EncodeRgbaBiasToTosaCustomPass(ARTIFACT_DIR / "rgba_bias"),
        ],
    )

    runtime = Runtime.get()
    if not runtime.backend_registry.is_available("VgfBackend"):
        raise RuntimeError("VgfBackend is not available in this build.")

    scale_add_model = ScaleAddModel().eval()
    scale_add_x = torch.linspace(-2.0, 2.0, steps=16, dtype=torch.float32).reshape(4, 4)
    scale_add_expected = scale_add_model(scale_add_x)

    scale_add_exported = export(scale_add_model, (scale_add_x,))
    scale_add_spec = VgfCompileSpec()
    scale_add_spec.dump_intermediate_artifacts_to(str(ARTIFACT_DIR / "scale_add"))
    scale_add_partitioner = VgfPartitioner(scale_add_spec)
    scale_add_partitioner.register_custom_partition_op(
        torch.ops.arm_example_custom_shader.scale_add.default
    )
    scale_add_edge_manager = to_edge_transform_and_lower(
        scale_add_exported,
        partitioner=[scale_add_partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    scale_add_exec_program = scale_add_edge_manager.to_executorch()
    scale_add_pte_path = ARTIFACT_DIR / "scale_add" / SCALE_ADD_PTE_NAME
    save_pte_program(scale_add_exec_program, str(scale_add_pte_path))

    scale_add_program = runtime.load_program(str(scale_add_pte_path))
    scale_add_method = scale_add_program.load_method("forward")
    assert scale_add_method is not None
    scale_add_actual = scale_add_method.execute((scale_add_x,))[0]

    if not torch.allclose(scale_add_expected, scale_add_actual, atol=1e-6, rtol=0.0):
        diff = (scale_add_expected - scale_add_actual).abs()
        raise AssertionError(
            f"Scale-add runtime mismatch. max_abs_diff={diff.max().item():.6f}"
        )

    rgba_bias_model = RgbaBiasModel().eval()
    rgba_bias_x = torch.arange(1.0, 61.0, dtype=torch.float32).reshape(1, 4, 3, 5)
    rgba_bias_expected = rgba_bias_model(rgba_bias_x)

    rgba_bias_exported = export(rgba_bias_model, (rgba_bias_x,))
    rgba_bias_spec = VgfCompileSpec()
    rgba_bias_spec.dump_intermediate_artifacts_to(str(ARTIFACT_DIR / "rgba_bias"))
    rgba_bias_partitioner = VgfPartitioner(rgba_bias_spec)
    rgba_bias_partitioner.register_custom_partition_op(
        torch.ops.arm_example_custom_shader.rgba_bias.default
    )
    rgba_bias_edge_manager = to_edge_transform_and_lower(
        rgba_bias_exported,
        partitioner=[rgba_bias_partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    rgba_bias_exec_program = rgba_bias_edge_manager.to_executorch()
    rgba_bias_pte_path = ARTIFACT_DIR / "rgba_bias" / RGBA_BIAS_PTE_NAME
    save_pte_program(rgba_bias_exec_program, str(rgba_bias_pte_path))

    rgba_bias_program = runtime.load_program(str(rgba_bias_pte_path))
    rgba_bias_method = rgba_bias_program.load_method("forward")
    assert rgba_bias_method is not None
    rgba_bias_actual = rgba_bias_method.execute((rgba_bias_x,))[0]

    if not torch.allclose(rgba_bias_expected, rgba_bias_actual, atol=1e-6, rtol=0.0):
        diff = (rgba_bias_expected - rgba_bias_actual).abs()
        raise AssertionError(
            f"RGBA image runtime mismatch. max_abs_diff={diff.max().item():.6f}"
        )

    print(f"Artifacts: {ARTIFACT_DIR.resolve()}")
    print("Scale-add input:")
    print(scale_add_x)
    print("Scale-add expected:")
    print(scale_add_expected)
    print("Scale-add runtime:")
    print(scale_add_actual)
    print("RGBA input:")
    print(rgba_bias_x)
    print("RGBA expected:")
    print(rgba_bias_expected)
    print("RGBA runtime:")
    print(rgba_bias_actual)
    print("Match: True")


# Helpers
def _ensure_glslc() -> str:
    glslc = shutil.which("glslc")
    if glslc is None:
        raise RuntimeError("`glslc` was not found on PATH.")
    return glslc


def _set_fake_tensor_meta(node: torch.fx.Node, value) -> None:
    node.meta["val"] = value
    if isinstance(value, list):
        if value:
            node.meta["tensor_meta"] = _extract_tensor_metadata(value[0])
    else:
        node.meta["tensor_meta"] = _extract_tensor_metadata(value)


def _compile_shader(
    output_dir: Path, shader_name: str, spirv_name: str, shader_source: str
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    shader_path = output_dir / shader_name
    spirv_path = output_dir / spirv_name
    shader_path.write_text(shader_source, encoding="utf-8")
    result = subprocess.run(  # nosec B603 - fixed trusted local tool
        [_ensure_glslc(), str(shader_path), "-o", str(spirv_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile {shader_path} with glslc.\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
    return base64.b64encode(spirv_path.read_bytes()).decode("ascii")


if __name__ == "__main__":
    main()
