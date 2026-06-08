# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import json
import shutil
import subprocess  # nosec B404 - required to invoke trusted local VGF dump tool
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backends.arm.test._custom_vgf_test_utils import (
    EncodeSamplerGridSampleToTosaCustomPass,
    EncodeTestAddToTosaCustomPass,
    EncodeThreesToTosaCustomPass,
    register_test_shader_library_ops,
    register_test_shader_partition_ops,
    register_test_threes_library_ops,
    register_test_threes_partition_ops,
    rewrite_aten_add_to_test_shader,
    rewrite_aten_grid_sample_to_test_shader,
)
from executorch.backends.arm._passes import RewriteMatmulPass
from executorch.backends.arm._passes.arm_pass_manager import (
    clear_registered_pass_insertions,
    register_pass_insertions_after,
)
from executorch.backends.arm.test.runner_utils import (
    arm_executor_runner_exists,
    get_elf_path,
    run_target,
    vkml_emulation_layer_installed,
)
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.backends.arm.vgf.model_converter import (
    find_model_converter_binary,
    model_converter_env,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.pass_base import ExportPass
from torch.export import export


def runtime_available() -> bool:
    return vkml_emulation_layer_installed() and arm_executor_runner_exists(
        "vkml_emulation_layer"
    )


def ensure_vgf_runtime() -> None:
    if not runtime_available():
        pytest.xfail("VGF runtime is not available on this system")


def ensure_glslc() -> None:
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")


@functools.lru_cache(maxsize=1)
def _model_converter_is_legacy_release() -> tuple[bool, str]:
    model_converter = find_model_converter_binary()
    if model_converter is None:
        warnings.warn(
            "Could not find model-converter while evaluating the VGF runtime "
            "legacy-version xfail gate; assuming a newer/custom build.",
            stacklevel=2,
        )
        return False, ""

    try:
        result = subprocess.run(  # nosec B603 - trusted local tool
            [model_converter, "--version"],
            check=True,
            capture_output=True,
            text=True,
            env=model_converter_env(),
        )
    except Exception as exc:
        warnings.warn(
            "Failed to query model-converter --version while evaluating the VGF "
            f"runtime legacy-version xfail gate ({exc}); assuming a newer/custom "
            "build.",
            stacklevel=2,
        )
        return False, ""

    version_text = (result.stdout or result.stderr).strip()
    if not version_text:
        warnings.warn(
            "model-converter --version returned no output while evaluating the VGF "
            "runtime legacy-version xfail gate; assuming a newer/custom build.",
            stacklevel=2,
        )
        return False, ""

    if "d8c1b8e" in version_text:
        return (
            True,
            "released model-converter build d8c1b8e predates required VGF custom "
            "shader features; use a newer source build",
        )

    warnings.warn(
        "model-converter legacy-version xfail gate expected d8c1b8e; detected "
        f"{version_text!r}. Assuming a newer/custom build.",
        stacklevel=2,
    )
    return False, ""


def xfail_if_legacy_model_converter_release() -> pytest.MarkDecorator:
    is_legacy_release, reason = _model_converter_is_legacy_release()
    return pytest.mark.xfail(is_legacy_release, reason=reason, strict=False)


def find_single_vgf_json(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("*.vgf.json"))
    if not matches:
        raise FileNotFoundError(f"No .vgf.json file found in {output_dir}")
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one .vgf.json file in {output_dir}, found {len(matches)}"
        )
    return matches[0]


def find_single_vgf_file(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("*.vgf"))
    if not matches:
        raise FileNotFoundError(f"No .vgf file found in {output_dir}")
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one .vgf file in {output_dir}, found {len(matches)}"
        )
    return matches[0]


def load_vgf_json(output_dir: Path) -> dict:
    try:
        vgf_json_path = find_single_vgf_json(output_dir)
    except FileNotFoundError as exc:
        if shutil.which("vgf_dump") is None:
            raise RuntimeError(
                f"No .vgf.json file found in {output_dir}, and `vgf_dump` was not "
                "found on PATH. `vgf_dump` is expected to be installed alongside "
                "`model_converter`; check that the model-converter tools are "
                "installed and available on PATH."
            ) from exc
        vgf_path = find_single_vgf_file(output_dir)
        vgf_json_path = vgf_path.with_suffix(vgf_path.suffix + ".json")
        subprocess.run(  # nosec B603, B607 - trusted local tool with fixed arguments
            ["vgf_dump", "-i", str(vgf_path), "-o", str(vgf_json_path)],
            check=True,
        )
    return json.loads(vgf_json_path.read_text())


def make_identity_grid(height: int, width: int) -> torch.Tensor:
    x_coords = (2.0 * (torch.arange(width, dtype=torch.float32) + 0.5) / width) - 1.0
    y_coords = (2.0 * (torch.arange(height, dtype=torch.float32) + 0.5) / height) - 1.0
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def make_input_tensor(height: int, width: int) -> torch.Tensor:
    xx = torch.arange(width, dtype=torch.float32).view(1, width).repeat(height, 1)
    yy = torch.arange(height, dtype=torch.float32).view(height, 1).repeat(1, width)
    c0 = xx + 10.0 * yy + 1.0
    c1 = 100.0 + xx
    c2 = 200.0 + yy
    c3 = torch.ones_like(xx)
    return torch.stack((c0, c1, c2, c3), dim=0).unsqueeze(0)


def make_sampler_probe_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    xx = torch.arange(8, dtype=torch.float32).view(1, 8).repeat(8, 1)
    yy = torch.arange(8, dtype=torch.float32).view(8, 1).repeat(1, 8)
    ramp = xx + 10.0 * yy + 1.0
    zeros = torch.zeros_like(ramp)
    ones = torch.ones_like(ramp)
    x = torch.stack((ramp, zeros, zeros, ones), dim=0).unsqueeze(0)
    x = x.contiguous(memory_format=torch.channels_last)

    coarse_x_pix = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=torch.float32
    )
    fine_x_pix = torch.linspace(0.26, 0.28, steps=9, dtype=torch.float32)
    y_pix = torch.tensor([3.0, 0.5, 0.0, 0.0], dtype=torch.float32)

    grid = torch.empty((1, y_pix.numel(), coarse_x_pix.numel(), 2), dtype=torch.float32)
    for row_idx, y_val in enumerate(y_pix.tolist()):
        x_positions = fine_x_pix if row_idx == 2 else coarse_x_pix
        grid[0, row_idx, :, 0] = (2.0 * x_positions + 1.0) / x.shape[-1] - 1.0
        grid[0, row_idx, :, 1] = (2.0 * y_val + 1.0) / x.shape[-2] - 1.0
    return x, grid


def execute_edge_manager(
    edge_mgr, example_inputs: tuple, output_dir: Path
) -> torch.Tensor:
    ensure_vgf_runtime()
    exec_prog = edge_mgr.to_executorch()
    outputs = run_target(
        exec_prog,
        example_inputs,
        output_dir,
        "vkml_emulation_layer",
        get_elf_path("vkml_emulation_layer"),
    )
    return outputs[0]


def lower_in_tree_vgf(module: torch.nn.Module, example_inputs: tuple, output_dir: Path):
    ensure_vgf_runtime()
    exported = export(module, example_inputs)
    expected = module(*example_inputs)
    vgf_spec = VgfCompileSpec()
    vgf_spec.dump_intermediate_artifacts_to(str(output_dir))
    partitioner = VgfPartitioner(vgf_spec)
    edge_mgr = to_edge_transform_and_lower(
        exported,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    actual = execute_edge_manager(edge_mgr, example_inputs, output_dir)
    return expected, actual, load_vgf_json(output_dir)


def _lower_custom_vgf(
    module: torch.nn.Module,
    example_inputs: tuple,
    output_dir: Path,
    *,
    use_add: bool = False,
    use_sampler: bool = False,
    use_threes: bool = False,
):
    ensure_vgf_runtime()
    ensure_glslc()
    if use_add or use_sampler:
        register_test_shader_library_ops()
    if use_threes:
        register_test_threes_library_ops()
    exported = export(module, example_inputs)
    if use_add:
        rewrite_aten_add_to_test_shader(exported.graph_module)
    if use_sampler:
        rewrite_aten_grid_sample_to_test_shader(exported.graph_module)
    expected = module(*example_inputs)
    vgf_spec = VgfCompileSpec()
    vgf_spec.dump_intermediate_artifacts_to(str(output_dir))
    partitioner = VgfPartitioner(vgf_spec)
    if use_add or use_sampler:
        register_test_shader_partition_ops(partitioner)
    if use_threes:
        register_test_threes_partition_ops(partitioner)
    clear_registered_pass_insertions()
    passes: list[ExportPass] = []
    if use_add:
        passes.append(EncodeTestAddToTosaCustomPass())
    if use_sampler:
        passes.append(EncodeSamplerGridSampleToTosaCustomPass())
    if use_threes:
        passes.append(EncodeThreesToTosaCustomPass())
    register_pass_insertions_after(RewriteMatmulPass, passes)
    try:
        edge_mgr = to_edge_transform_and_lower(
            exported,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    finally:
        clear_registered_pass_insertions()
    actual = execute_edge_manager(edge_mgr, example_inputs, output_dir)
    return expected, actual, load_vgf_json(output_dir)


def lower_add_vgf(module: torch.nn.Module, example_inputs: tuple, output_dir: Path):
    return _lower_custom_vgf(
        module,
        example_inputs,
        output_dir,
        use_add=True,
    )


def lower_sampler_vgf(module: torch.nn.Module, example_inputs: tuple, output_dir: Path):
    return _lower_custom_vgf(
        module,
        example_inputs,
        output_dir,
        use_sampler=True,
    )


def lower_add_and_sampler_vgf(
    module: torch.nn.Module, example_inputs: tuple, output_dir: Path
):
    return _lower_custom_vgf(
        module,
        example_inputs,
        output_dir,
        use_add=True,
        use_sampler=True,
    )


def lower_sampler_and_threes_vgf(
    module: torch.nn.Module, example_inputs: tuple, output_dir: Path
):
    return _lower_custom_vgf(
        module,
        example_inputs,
        output_dir,
        use_sampler=True,
        use_threes=True,
    )


def lower_threes_vgf(module: torch.nn.Module, example_inputs: tuple, output_dir: Path):
    return _lower_custom_vgf(
        module,
        example_inputs,
        output_dir,
        use_threes=True,
    )


def alias_groups(vgf_json: dict) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for resource in vgf_json.get("resources", []):
        alias_group_id = resource.get("alias_group_id")
        if alias_group_id is None:
            continue
        groups.setdefault(int(alias_group_id), []).append(resource)
    return groups


def segment_types(vgf_json: dict) -> list[str]:
    return [segment["type"] for segment in vgf_json["model_sequence"]["segments"]]
