# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from executorch.backends.aoti import aoti_backend
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes import helion_cute, lower_helion
from executorch.backends.cuda.passes.lower_helion import LowerHelionPass
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph
from torch.fx.experimental.proxy_tensor import make_fx
from triton.runtime.jit import JITFunction

try:
    import helion
    import helion.language as hl
except ImportError as error:
    # Helion is an optional dependency of the CUDA backend.
    raise unittest.SkipTest(f"Helion is not installed: {error}") from error

# Outside the guard on purpose: Helion being present but its private capture
# modules having moved must fail this test, not skip the whole file.
from helion._compiler._dynamo.higher_order_ops import (
    helion_kernel_side_table,
    helion_kernel_wrapper_functional,
)
from helion.runtime.kernel import BoundKernel


def _run_twice(program, inputs: torch.Tensor) -> list[list[np.ndarray]]:
    """Run forward twice; each run returns all of the method's outputs."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "model.pte").write_bytes(program.buffer)
        program.write_tensor_data_to_file(directory)
        runner = os.environ.get("RUN_CUDA_PROGRAM")
        if runner is None:
            # The pip package's Python runtime includes the CUDA backend.
            from executorch.runtime import Runtime

            method = (
                Runtime.get()
                .load_program(root / "model.pte", data_path=root / "aoti_cuda_blob.ptd")
                .load_method("forward")
            )
            return [
                [output.cpu().numpy() for output in method.execute([inputs.cpu()])]
                for _ in range(2)
            ]
        # In Buck, LibTorch and the CUDA shims define the same aoti_torch_*
        # symbols, so the program runs in a LibTorch-free process.
        np.save(root / "input.npy", inputs.cpu().numpy())
        np.save(root / "input2.npy", inputs.cpu().numpy())
        subprocess.run([runner, directory], check=True, timeout=120)

        def load_outputs(prefix: str) -> list[np.ndarray]:
            outputs = [np.load(root / f"{prefix}.npy")]
            index = 1
            while (root / f"{prefix}_{index}.npy").exists():
                outputs.append(np.load(root / f"{prefix}_{index}.npy"))
                index += 1
            return outputs

        return [load_outputs("output"), load_outputs("output2")]


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result[tile] = x[tile] + y[tile]
    return result


def _increment(x: torch.Tensor) -> None:
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + 1


def _increment_and_double(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + 1
        result[tile] = x[tile] * 2
    return result


def _increment_and_return(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + 1
    return x


def _add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        result[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return result


def _increment_2d(x: torch.Tensor) -> None:
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        x[tile_m, tile_n] = x[tile_m, tile_n] + 1


def _add3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result[tile] = x[tile] + y[tile] + z[tile]
    return result


def _scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result[tile] = x[tile] * scale
    return result


def _two_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(x)
    second = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        first[tile] = x[tile] + 1
        second[tile] = x[tile] * 2
    return first, second


def _attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    """Flash attention with an online softmax, after Helion's attention example."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim) * 1.44269504  # log2(e)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(n_dim):
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q * qk_scale, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            p = torch.exp2(qk - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            v = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc * alpha[:, :, None], p.to(v.dtype), v)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view(q_in.size())


def _depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Valid depthwise 1-D convolution: x [C, T + K - 1], weight [C, K]."""
    channels = x.size(0)
    width = hl.specialize(weight.size(1))
    length = x.size(1) - width + 1
    out = torch.empty([channels, length], dtype=x.dtype, device=x.device)
    for tile_c, tile_t in hl.tile([channels, length]):
        acc = hl.zeros([tile_c, tile_t], dtype=torch.float32)
        for k in hl.static_range(width):
            window = hl.load(x, [tile_c, tile_t.index + k])
            acc = acc + window.to(torch.float32) * weight[tile_c, k][:, None]
        out[tile_c, tile_t] = acc.to(out.dtype)
    return out


class HelionLaunchContractTest(unittest.TestCase):
    def test_accepts_ordinary_launch(self) -> None:
        source = "def add(x):\n    _launcher(k, (1,), x, num_warps=4, num_stages=1, launch_cooperative_grid=False)\n"
        self.assertIsNone(lower_helion._check_single_launch_site(source, "add"))

    def test_rejects_unsupported_launch_options(self) -> None:
        for option in (
            "launch_cooperative_grid=True",
            "num_ctas=2",
            "launch_pdl=True",
            "_persistent_state_specs=states",
            "_minimum_resident_programs=4",
            "**options",
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(NotImplementedError, "launch option"):
                    lower_helion._check_single_launch_site(
                        f"def add(x):\n    _launcher(k, (1,), x, {option})\n", "add"
                    )

    def test_rejects_missing_or_multiple_launches(self) -> None:
        for body in ("return x", "_launcher(k, (1,), x)\n    _launcher(k, (1,), x)"):
            with self.subTest(body=body):
                with self.assertRaisesRegex(
                    NotImplementedError, "single Triton launch"
                ):
                    lower_helion._check_single_launch_site(
                        f"def add(x):\n    {body}\n", "add"
                    )
        with self.assertRaisesRegex(ValueError, "entrypoint"):
            lower_helion._check_single_launch_site(
                "def other(x):\n    return x\n", "add"
            )

    def test_binds_helion_jit_helpers_to_global_names(self) -> None:
        source = (
            "from helion.runtime.triton import helpers as helion_triton_helpers\n"
            "def f(x):\n"
            "    return helion_triton_helpers.max_propagate_nan(x, 0)\n"
        )
        bound = lower_helion._bind_helion_jit_helpers(source)
        self.assertIn(
            "max_propagate_nan = helion_triton_helpers.max_propagate_nan", bound
        )
        self.assertIn("return max_propagate_nan(x, 0)", bound)

    def test_rejects_old_helion_version(self) -> None:
        with mock.patch.object(
            lower_helion.importlib.metadata,
            "version",
            return_value="1.5.1",
        ):
            with self.assertRaisesRegex(RuntimeError, "Helion 1.5.2 or newer"):
                lower_helion._check_helion_version()
            # The check is wired into capture-module loading, not only a
            # standalone helper nobody calls.
            lower_helion._helion_capture_modules.cache_clear()
            try:
                with self.assertRaisesRegex(RuntimeError, "Helion 1.5.2 or newer"):
                    lower_helion._helion_capture_modules()
            finally:
                lower_helion._helion_capture_modules.cache_clear()

    def test_rejects_placeholder_mutation(self) -> None:
        graph = Graph()
        x, y = graph.placeholder("x"), graph.placeholder("y")
        copied = graph.call_function(torch.ops.aten.copy_.default, (x, y))
        graph.output((copied,))
        traced = torch.fx.GraphModule({}, graph)
        with self.assertRaisesRegex(NotImplementedError, "write to an input"):
            lower_helion._check_no_placeholder_mutation(traced, "add")

    def test_rejects_helper_name_clash(self) -> None:
        source = (
            "from helion.runtime.triton import helpers as helion_triton_helpers\n"
            "max_propagate_nan = 0\n"
            "def f(x):\n"
            "    return helion_triton_helpers.max_propagate_nan(x, 0)\n"
        )
        with self.assertRaisesRegex(NotImplementedError, "clashes"):
            lower_helion._bind_helion_jit_helpers(source)

    def test_rejects_non_getitem_output_use(self) -> None:
        graph = Graph()
        hop, other = graph.placeholder("hop"), graph.placeholder("other")
        graph.call_function(torch.ops.aten.add.Tensor, (hop, other))
        with self.assertRaisesRegex(NotImplementedError, "accessed by getitem"):
            lower_helion._reroute_hop_outputs(graph, hop, (other,), None)

    def _captured(self, **overrides) -> lower_helion._CapturedHelionCall:
        kernel = SimpleNamespace(
            settings=SimpleNamespace(backend="triton", static_shapes=True),
            configs=[{}],
            signature=SimpleNamespace(parameters={}),
        )
        fields = {
            "kernel": kernel,
            "is_functional": False,
            "tensor_args": {},
            "has_constant_args": False,
            "tensors_to_clone": (),
            "tensor_output_count": 0,
            "aliases_inputs": False,
            "mutates_inputs": False,
            "output_spec": {},
        }
        fields.update(overrides)
        return lower_helion._CapturedHelionCall(**fields)

    def test_rejects_container_argument(self) -> None:
        captured = self._captured(tensor_args={"xs": [None]})
        captured.kernel.signature.parameters = {"xs": None}
        with self.assertRaisesRegex(NotImplementedError, "tensor-only arguments"):
            lower_helion._parse_helion_call(captured)

    def test_rejects_aliased_outputs(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "aliased outputs"):
            lower_helion._parse_helion_call(self._captured(aliases_inputs=True))

    def test_rejects_unfunctionalized_mutation(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "functionalized"):
            lower_helion._parse_helion_call(self._captured(mutates_inputs=True))

    def test_rejects_host_tensor_descriptors(self) -> None:
        captured = self._captured()
        captured.kernel.configs = [{"host_tensor_descriptors": True}]
        call = lower_helion._HelionCall(captured, [], (), ())
        with self.assertRaisesRegex(NotImplementedError, "host tensor descriptors"):
            lower_helion._render_entrypoint(call)

    def test_rejects_unsupported_backend(self) -> None:
        kernel = SimpleNamespace(
            settings=SimpleNamespace(backend="pallas", static_shapes=True)
        )
        captured = lower_helion._CapturedHelionCall(
            kernel=kernel,
            is_functional=False,
            tensor_args={},
            has_constant_args=False,
            tensors_to_clone=(),
            tensor_output_count=1,
            aliases_inputs=False,
            mutates_inputs=False,
            output_spec={},
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "only the Triton and CuTe backends; this kernel uses 'pallas'",
        ):
            lower_helion._parse_helion_call(captured)

    def test_cuda_backend_runs_helion_lowering_last(self) -> None:
        passes = CudaBackend.get_custom_passes([])
        self.assertIsInstance(passes[-1], LowerHelionPass)
        windows = CudaBackend.get_custom_passes([CompileSpec("platform", b"windows")])
        self.assertEqual(windows[-1].target_platform, "windows")

    def test_ordinary_graph_does_not_import_helion(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        program = torch.export.export(Model(), (torch.ones(4),), strict=True)
        original = program.graph_module.code
        # An earlier test in this process may have cached the Helion modules;
        # clear the cache so consulting Helion here would re-import it.
        lower_helion._helion_capture_modules.cache_clear()
        with mock.patch.object(
            lower_helion.importlib,
            "import_module",
            side_effect=AssertionError("unexpected Helion import"),
        ):
            LowerHelionPass()(program)
        self.assertEqual(program.graph_module.code, original)


class HelionCuteContractTest(unittest.TestCase):
    """GPU-free tests of the CuTe export helpers OSS CI can run anywhere."""

    def test_xyz_pads_and_rejects_dynamic_dimensions(self) -> None:
        self.assertEqual(helion_cute._xyz((2,)), (2, 1, 1))
        self.assertEqual(helion_cute._xyz((2, 3, 4)), (2, 3, 4))
        with self.assertRaisesRegex(NotImplementedError, "static launch dimensions"):
            helion_cute._xyz((2.0,))

    def test_capture_launch_contract(self) -> None:
        x = torch.zeros(4)
        kernel = object()

        def entrypoint(t, _launcher):
            return _launcher(kernel, (4,), t)

        captured = helion_cute._capture_launch(entrypoint, "_launcher", (x,))
        self.assertIs(captured.cute_kernel, kernel)
        self.assertEqual(captured.grid, (4, 1, 1))
        self.assertEqual(captured.block, helion_cute._DEFAULT_BLOCK)
        self.assertEqual(captured.tensors, (x,))

        def two_launches(t, _launcher):
            _launcher(kernel, (4,), t)
            _launcher(kernel, (4,), t)

        with self.assertRaisesRegex(NotImplementedError, "single launch"):
            helion_cute._capture_launch(two_launches, "_launcher", (x,))

        def extra_kwarg(t, _launcher):
            return _launcher(kernel, (4,), t, num_warps=4)

        with self.assertRaisesRegex(NotImplementedError, "launch options"):
            helion_cute._capture_launch(extra_kwarg, "_launcher", (x,))

        def scalar_arg(t, _launcher):
            return _launcher(kernel, (4,), t, 1)

        with self.assertRaisesRegex(NotImplementedError, "tensor arguments only"):
            helion_cute._capture_launch(scalar_arg, "_launcher", (x,))

        def empty_grid(t, _launcher):
            return _launcher(kernel, (0,), t)

        with self.assertRaisesRegex(NotImplementedError, "empty grid"):
            helion_cute._capture_launch(empty_grid, "_launcher", (x,))

    def test_shim_template_renders(self) -> None:
        shim = helion_cute._SHIM_TEMPLATE.format(
            prefix="helion_cute_x",
            shim="aoti_torch_cuda_helion_cute_x",
            parameters="AtenTensorHandle t0",
            count=1,
            ranks="1",
            layout_arrays=(
                "constexpr int64_t kSizes0[] = {4};\n"
                "constexpr int64_t kStrides0[] = {1};"
            ),
            sizes="kSizes0",
            strides="kStrides0",
            handles="t0",
            alignments="16",
            pointer_args="pointers[0]",
            grid="4, 1, 1",
        )
        self.assertIn("aoti_torch_cuda_helion_cute_x(AtenTensorHandle t0", shim)
        self.assertIn("kAlignments[kCount] = {16}", shim)
        self.assertIn("cute_dsl_helion_cute_x_wrapper", shim)
        # Compile the rendered shim against a stub of the exported header
        # when a CUDA toolkit is available, so template breakage is caught
        # without a Blackwell GPU.
        from torch._inductor.cpp_builder import get_cpp_compiler
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            self.skipTest("no CUDA toolkit to compile the rendered shim")
        stub = (
            "#pragma once\n#include <cstdint>\n"
            "struct helion_cute_x_Kernel_Module_t { int unused; };\n"
            "static void helion_cute_x_Kernel_Module_Load(\n"
            "    helion_cute_x_Kernel_Module_t*) {}\n"
            "static int32_t cute_dsl_helion_cute_x_wrapper(\n"
            "    helion_cute_x_Kernel_Module_t*, void*, int, int, int,\n"
            "    cudaStream_t) { return 0; }\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helion_cute_x.h").write_text(stub)
            (root / "shim.cpp").write_text(shim)
            subprocess.run(
                [
                    get_cpp_compiler(),
                    "-c",
                    "-fPIC",
                    "-O2",
                    "-std=c++17",
                    f"-I{root}",
                    f"-I{CUDA_HOME}/include",
                    str(root / "shim.cpp"),
                    "-o",
                    str(root / "shim.o"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_runtime_archive_builds_and_defines_forwards(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            self.skipTest("CUDA toolkit required")
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(helion_cute, "cache_dir", return_value=directory),
                mock.patch.object(helion_cute, "_RUNTIME_ARCHIVE_DIRECTORY", None),
            ):
                runtime_directory = helion_cute._runtime_archive_directory()
            archive = runtime_directory / f"lib{helion_cute._RUNTIME_LIBRARY}.a"
            self.assertTrue(archive.is_file())
            defined = subprocess.run(
                ["nm", "--defined-only", str(archive)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for symbol in (
                "_cudaLibraryLoadData",
                "_cudaLaunchKernelEx",
                "_cudaKernelSetAttributeForDevice",
                "_cuTensorMapEncodeTiled",
                "cuda_dialect_assert_success",
            ):
                self.assertIn(symbol, defined)

    def test_example_like_offsets(self) -> None:
        # Aligned arguments are compiled at their allocation; everything
        # else gets an example one element into its storage, so only
        # element alignment is assumed. LowerHelionPass keeps root
        # operands whose offset is not 16-byte aligned out of the aligned
        # set, so a view parameter takes the second branch.
        tensor = torch.empty(257)
        self.assertEqual(
            helion_cute._example_like(tensor, aligned=True).storage_offset(), 0
        )
        self.assertEqual(
            helion_cute._example_like(tensor, aligned=False).storage_offset(), 1
        )

    def test_link_exported_launches_clears_and_remembers(self) -> None:
        helion_cute.begin_program()
        helion_cute._STATE.program_launches()["p"] = helion_cute._ExportedLaunch(
            op=torch.ops.aten.add.Tensor,
            shim_declaration="AOTITorchError aoti_torch_cuda_p();",
            library="p",
            library_directory=Path("/tmp"),
        )
        with helion_cute.link_exported_launches():
            self.assertIn("p", helion_cute._STATE.program_launches())
        self.assertEqual(helion_cute._STATE.program_launches(), {})
        self.assertTrue(helion_cute.last_program_had_launches())
        helion_cute.begin_program()
        self.assertFalse(helion_cute.last_program_had_launches())

    def test_multiple_launches_recorded_for_linking(self) -> None:
        def fake_launch(library: str, directory: Path) -> helion_cute._ExportedLaunch:
            (directory / f"lib{library}.a").touch()
            return helion_cute._ExportedLaunch(
                op=torch.ops.aten.add.Tensor,
                shim_declaration="",
                library=library,
                library_directory=directory,
            )

        self.addCleanup(helion_cute.begin_program)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dir_a, dir_b = root / "a", root / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            helion_cute.record_program_launch(fake_launch("launch_a", dir_a))
            helion_cute.begin_program()
            self.assertEqual(helion_cute.current_program_launches(), [])
            helion_cute.record_program_launch(fake_launch("launch_a", dir_a))
            helion_cute.record_program_launch(fake_launch("launch_b", dir_b))
            self.assertEqual(
                {e.library for e in helion_cute.current_program_launches()},
                {"launch_a", "launch_b"},
            )
            with mock.patch.object(
                helion_cute,
                "_runtime_archive_directory",
                return_value=root / "runtime",
            ):
                directories, libraries = helion_cute.program_link_options()
            self.assertEqual(
                libraries,
                ["launch_a", "launch_b", helion_cute._RUNTIME_LIBRARY],
            )
            self.assertEqual(libraries.count(helion_cute._RUNTIME_LIBRARY), 1)
            self.assertEqual(
                directories, [str(dir_a), str(dir_b), str(root / "runtime")]
            )

    def test_fallback_kernels_cover_program_launches(self) -> None:
        fake = helion_cute._ExportedLaunch(
            op=torch.ops.aten.add.Tensor,
            shim_declaration="AOTITorchError aoti_torch_cuda_q();",
            library="q_probe",
            library_directory=Path("/tmp"),
        )
        helion_cute.begin_program()
        try:
            self.assertNotIn(
                "helion_cute::q_probe", helion_cute.exported_fallback_kernels()
            )
            helion_cute._STATE.program_launches()["q_probe"] = fake
            self.assertIn(
                "helion_cute::q_probe", helion_cute.exported_fallback_kernels()
            )
        finally:
            helion_cute.begin_program()

    def test_compile_and_export_rejections(self) -> None:
        tensor = torch.zeros(2)
        captured = helion_cute._CapturedLaunch(
            cute_kernel=object(),
            grid=(1, 1, 1),
            block=(1, 1, 1),
            compile_options=None,
            tensors=(tensor,),
        )

        def fake_modules(build_result, stage=False):
            launcher = SimpleNamespace(
                _cute_stage_under_aligned_operands=lambda k, t: stage,
                _build_cute_schema_and_args=lambda k, t, g: build_result,
            )
            arg_signature = SimpleNamespace(schema_pointer_alignment=lambda entry: 16)
            return launcher, arg_signature, object(), object()

        valid = SimpleNamespace(
            schema=(("tensor", 0, 0, 0, 0),),
            owned_tensors=[],
            launch_args=(tensor, 1, 1, 1),
        )
        cases = (
            (fake_modules(valid, stage=True), "under-aligned"),
            (
                fake_modules(
                    SimpleNamespace(
                        schema=valid.schema,
                        owned_tensors=[tensor],
                        launch_args=valid.launch_args,
                    )
                ),
                "workspace",
            ),
            (
                fake_modules(
                    SimpleNamespace(
                        schema=(("scalar", 0, 0, 0, 0),),
                        owned_tensors=[],
                        launch_args=valid.launch_args,
                    )
                ),
                "static tensor arguments only",
            ),
            (
                fake_modules(
                    SimpleNamespace(
                        schema=valid.schema,
                        owned_tensors=[],
                        launch_args=(tensor,),
                    )
                ),
                "unexpected launch ABI",
            ),
        )
        for modules, message in cases:
            with self.subTest(message=message):
                with mock.patch.object(
                    helion_cute, "_cute_launcher_modules", return_value=modules
                ):
                    with tempfile.TemporaryDirectory() as directory:
                        with self.assertRaisesRegex(NotImplementedError, message):
                            helion_cute._compile_and_export(
                                captured, "probe", Path(directory)
                            )

    def test_error_check_macro_guard(self) -> None:
        helion_cute._check_error_check_macro(
            "#ifndef CUTE_DSL_CUDA_ERROR_CHECK\n"
            "#define CUTE_DSL_CUDA_ERROR_CHECK(err) (void)(err)\n"
            "#endif\n"
            "CUTE_DSL_CUDA_ERROR_CHECK(load());\n"
        )
        with self.assertRaisesRegex(NotImplementedError, "never uses"):
            helion_cute._check_error_check_macro("int placeholder;\n")
        with self.assertRaisesRegex(NotImplementedError, "#ifndef guard"):
            helion_cute._check_error_check_macro(
                "#define CUTE_DSL_CUDA_ERROR_CHECK(err) (void)(err)\n"
                "CUTE_DSL_CUDA_ERROR_CHECK(load());\n"
            )

    def test_register_op_supports_more_than_26_arguments(self) -> None:
        op = helion_cute._register_op(
            "many_args_probe", [i == 29 for i in range(30)], lambda tensors: None
        )
        self.assertEqual(len(op._schema.arguments), 30)
        self.assertIn("a29!", str(op._schema))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LowerHelionTest(unittest.TestCase):
    """End-to-end and pass tests, run for each Helion backend the pass lowers."""

    BACKEND = "triton"
    # Names the node the pass leaves for a kernel launch of BACKEND.
    LAUNCH_MARKER = "triton_kernel_wrapper"

    def setUp(self) -> None:
        # Tests that run the pass without compiling would otherwise leak
        # recorded CuTe launches into later fallback-kernel assertions.
        self.addCleanup(helion_cute.begin_program)

    def _kernel(self, fn, config, static_shapes=True):
        return helion.kernel(
            backend=self.BACKEND, config=config, static_shapes=static_shapes
        )(fn)

    def _has_launch(self, graph) -> bool:
        # A launch that writes its arguments may be functionalized into a node
        # whose first argument is the launch op.
        return any(
            self.LAUNCH_MARKER in str(n.target)
            or (n.args and self.LAUNCH_MARKER in str(n.args[0]))
            for n in graph.nodes
        )

    def _export_and_run(
        self, model, inputs, expected, expected_second=None, rtol=1e-5, atol=1e-6
    ):
        exported = torch.export.export(model, (inputs,), strict=True)
        self.assertTrue(
            any(
                "helion_kernel_wrapper" in str(n.target)
                for module in exported.graph_module.modules()
                if isinstance(module, torch.fx.GraphModule)
                for n in module.graph.nodes
            )
        )
        with mock.patch.object(
            lower_helion,
            "_trace_helion_call",
            wraps=lower_helion._trace_helion_call,
        ) as trace:
            edge = to_edge_transform_and_lower(
                exported,
                partitioner=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
            )
        self.assertGreater(
            trace.call_count, 0, "conversion must run inside CUDA preprocessing"
        )
        self.assertIn(
            "executorch_call_delegate", edge.exported_program().graph_module.code
        )
        first, second = _run_twice(edge.to_executorch(), inputs)
        second_expected = expected if expected_second is None else expected_second
        for actual_outputs, values in ((first, expected), (second, second_expected)):
            expected_outputs = (
                values if isinstance(values, (tuple, list)) else (values,)
            )
            self.assertEqual(len(actual_outputs), len(expected_outputs))
            for actual, value in zip(actual_outputs, expected_outputs):
                np.testing.assert_allclose(
                    actual.astype(np.float32),
                    value.float().cpu().numpy(),
                    rtol=rtol,
                    atol=atol,
                )

    def test_attention_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _attention,
            helion.Config(block_sizes=[1, 64, 32], num_warps=4, num_stages=2),
            static_shapes=True,
        )
        qkv = torch.randn(3, 2, 4, 256, 64, device="cuda", dtype=torch.float16)

        class Model(torch.nn.Module):
            def forward(self, qkv):
                return kernel(qkv[0], qkv[1], qkv[2])

        expected = torch.nn.functional.scaled_dot_product_attention(
            qkv[0].float(), qkv[1].float(), qkv[2].float()
        )
        self._export_and_run(Model(), qkv, expected, rtol=2e-2, atol=2e-2)

    def test_depthwise_conv1d_with_weight_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _depthwise_conv1d, helion.Config(block_sizes=[16, 64]), static_shapes=True
        )
        x = torch.randn(64, 515, device="cuda")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                weight = torch.randn(64, 4, device="cuda")
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def forward(self, x):
                return kernel(x, self.weight)

        model = Model()
        expected = torch.nn.functional.conv1d(
            x[None], model.weight[:, None, :], groups=64
        )[0]
        self._export_and_run(model, x, expected, rtol=1e-4, atol=1e-4)

    def test_native_model_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        inputs = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x[0], x[1]) * 2

        self._export_and_run(Model(), inputs, (inputs[0] + inputs[1]) * 2)
        torch.testing.assert_close(Model()(inputs), (inputs[0] + inputs[1]) * 2)

    def test_split_view_reads_after_pass(self) -> None:
        # The pieces of a restored split must be specialized on their
        # real row stride. This runs the pass directly, because split
        # cannot be preserved through to_edge at all.
        kernel = self._kernel(
            _add_2d, helion.Config(block_sizes=[16, 64]), static_shapes=True
        )
        x = torch.randn(64, 256, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                gate, up = y.split(128, dim=1)
                return kernel(gate, up)

        program = torch.export.export(Model(), (x,), strict=True)
        # Simulate the stale fakes ReplaceViewCopyWithViewPass leaves:
        # without the refresh this test reads the wrong elements.
        for node in program.graph_module.graph.nodes:
            if node.op == "call_function" and "split" in str(node.target):
                node.meta["val"] = [piece.contiguous() for piece in node.meta["val"]]
            if node.op == "call_function" and node.target is operator.getitem:
                node.meta["val"] = node.meta["val"].contiguous()
        LowerHelionPass()(program)
        torch.testing.assert_close(program.module()(x), x[:, :128] + x[:, 128:])

    def test_downstream_of_view_reads_after_pass(self) -> None:
        # An op computed from a restored view inherits the stale layout
        # too; the refresh must propagate through it.
        kernel = self._kernel(
            _add_2d, helion.Config(block_sizes=[16, 64]), static_shapes=True
        )
        x = torch.randn(256, 64, device="cuda").t()

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                return kernel(y[:, :128].clone(), y[:, 128:].clone())

        program = torch.export.export(Model(), (x,), strict=True)
        # Corrupt only the slices and the clones computed from them, the
        # state the view-copy rewrite leaves behind; everything else
        # keeps its real (transposed) layout.
        for node in program.graph_module.graph.nodes:
            if node.op != "call_function" or "val" not in node.meta:
                continue
            target = str(node.target)
            if "slice" in target or "clone" in target:
                node.meta["val"] = node.meta["val"].contiguous()
        lower_helion._refresh_view_fakes(program.graph_module)
        # The clones' fakes must have been recomputed from the refreshed
        # slices: with stale contiguous fakes the kernel would be
        # specialized on row-major strides.
        clones = [
            node
            for node in program.graph_module.graph.nodes
            if node.op == "call_function" and "clone" in str(node.target)
        ]
        self.assertTrue(clones)
        for clone in clones:
            self.assertEqual(clone.meta["val"].stride(), (1, 64))

    def test_nested_cond_on_view_operand_export_and_cpp_runtime(self) -> None:
        # A cond nested in a cond branch, on a restored slice operand:
        # the fake refresh must reach the inner branch placeholders.
        kernel = self._kernel(
            _add_2d, helion.Config(block_sizes=[16, 64]), static_shapes=True
        )
        x = torch.rand(64, 256, device="cuda") + 0.5

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                return torch.cond(
                    x.sum() > 0,
                    lambda v: torch.cond(
                        v.sum() > 0, lambda v: kernel(v, v), lambda v: v.neg(), (v,)
                    ),
                    lambda v: v.neg(),
                    (y[:, :128],),
                )

        self._export_and_run(Model(), x, x[:, :128] * 2)

    def test_cond_output_reads_after_pass(self) -> None:
        # The Helion call reads a cond output: the refreshed branch
        # layouts must reach the cond's getitem users, or the kernel is
        # specialized on contiguous strides. The stale state is the one
        # the view-copy rewrite leaves behind, reproduced here by
        # corrupting the fakes computed from the transposed slice.
        kernel = helion.kernel(
            config=helion.Config(block_sizes=[16, 64]), static_shapes=True
        )(_add_2d)
        x = (torch.rand(256, 64, device="cuda") + 0.5).t()

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                chosen = torch.cond(
                    x.sum() > 0,
                    lambda v: v + 1,
                    lambda v: v * 3,
                    (y[:, :128],),
                )
                return kernel(chosen, chosen)

        program = torch.export.export(Model(), (x,), strict=True)
        for module in program.graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function" or "val" not in node.meta:
                    continue
                value = node.meta["val"]
                values = value if isinstance(value, (list, tuple)) else [value]
                if any(
                    isinstance(v, torch.Tensor) and tuple(v.shape) == (64, 128)
                    for v in values
                ):
                    node.meta["val"] = (
                        [
                            v.contiguous() if isinstance(v, torch.Tensor) else v
                            for v in values
                        ]
                        if isinstance(value, (list, tuple))
                        else value.contiguous()
                    )
        LowerHelionPass()(program)
        torch.testing.assert_close(program.module()(x), (x[:, :128] + 1) * 2)

    def test_rejects_helion_call_on_while_loop_output(self) -> None:
        # A while_loop output's layout cannot be recomputed by
        # replaying it, so a Helion call on it is rejected instead of
        # being specialized on the stale fake. The stale state is the
        # one the view-copy rewrite leaves behind, reproduced here by
        # corrupting the slice fake the loop carries.
        kernel = helion.kernel(
            config=helion.Config(block_sizes=[16, 64]), static_shapes=True
        )(_add_2d)
        x = torch.randn(256, 64, device="cuda").t()

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1

                def cond_fn(i, acc):
                    return i < 2

                def body_fn(i, acc):
                    return i + 1, acc + 0

                _, out = torch.while_loop(cond_fn, body_fn, (0, y[:, :128]))
                return kernel(out, out)

        program = torch.export.export(Model(), (x,), strict=True)
        for node in program.graph_module.graph.nodes:
            if node.op == "call_function" and "slice" in str(node.target):
                node.meta["val"] = node.meta["val"].contiguous()
        with self.assertRaisesRegex(
            NotImplementedError, "cannot recompute the layout of an argument"
        ):
            LowerHelionPass()(program)

    def test_rejects_helion_call_in_while_loop(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                def cond_fn(i, acc):
                    return i < 2

                def body_fn(i, acc):
                    return i + 1, kernel(acc, acc)

                _, out = torch.while_loop(cond_fn, body_fn, (0, x))
                return out

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(
            NotImplementedError, "nested control flow other than torch.cond"
        ):
            LowerHelionPass()(program)

    def test_refresh_replay_failure_raises(self) -> None:
        kernel = self._kernel(
            _add_2d, helion.Config(block_sizes=[16, 64]), static_shapes=True
        )
        x = torch.randn(64, 256, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                gate, up = y.split(128, dim=1)
                return kernel(gate, up)

        program = torch.export.export(Model(), (x,), strict=True)
        for node in program.graph_module.graph.nodes:
            if node.op == "call_function" and "split" in str(node.target):
                # Corrupt the split input's fake so replaying the split
                # on it fails loudly instead of keeping a stale layout.
                source = node.args[0]
                source.meta["val"] = source.meta["val"].reshape(-1)
        with self.assertRaisesRegex(RuntimeError, "cannot recompute the layout"):
            LowerHelionPass()(program)

    def test_strided_view_reads_export_and_cpp_runtime(self) -> None:
        # Read-only strided views must be specialized on their real
        # strides, not the contiguous fakes the view-copy rewrite leaves
        # behind.
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(514, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                return kernel(y[::2], y[1::2])

        self._export_and_run(Model(), x, x[::2] + x[1::2])

    def test_multiple_calls_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        inputs = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                first = kernel(x[0], x[1])
                return kernel(first, x[1])

        self._export_and_run(Model(), inputs, inputs[0] + 2 * inputs[1])

    def test_multiple_outputs_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _two_outputs, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                first, second = kernel(x)
                # Subtraction, not addition, so swapped outputs fail.
                return first - second

        self._export_and_run(Model(), x, (x + 1) - x * 2)

    @unittest.skipIf(
        # Only the internal Buck run sets RUN_CUDA_PROGRAM; its SM80
        # workers take an unsupported TLX matmul path. Other SM80 runners
        # still exercise this case.
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0)
        and os.environ.get("RUN_CUDA_PROGRAM") is not None,
        "the matmul consumer takes an unsupported TLX path on internal SM80",
    )
    def test_padded_intermediate_export_and_cpp_runtime(self) -> None:
        # The Helion kernel receives the 2-D intermediate itself, whose row
        # is unaligned, and a matmul consumes it too: if Inductor pads its
        # strides after the pass specialized the kernel to the exported
        # strides, the result is silently wrong, so this must keep
        # matching eager.
        kernel = self._kernel(
            _add_2d, helion.Config(block_sizes=[32, 256]), static_shapes=True
        )
        x = torch.randn(64, 1500, device="cuda")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                weight = torch.randn(1500, 8, device="cuda")
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def forward(self, x):
                y = x * 2
                return kernel(y, y), y @ self.weight

        model = Model()
        y = x * 2
        self._export_and_run(model, x, (y * 2, y @ model.weight), rtol=1e-3, atol=1e-3)

    def test_two_kernels_export_and_cpp_runtime(self) -> None:
        add = self._kernel(_add, helion.Config(block_sizes=[256]), static_shapes=True)
        add3 = self._kernel(_add3, helion.Config(block_sizes=[256]), static_shapes=True)
        x = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                summed = add(x[0], x[1])
                return add3(summed, x[0], x[1])

        self._export_and_run(Model(), x, 2 * (x[0] + x[1]))

    def test_mutation_and_new_output_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _increment_and_double, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                working = x.clone()
                doubled = kernel(working)
                # Distinct coefficients, so swapping the kernel's new
                # output with its mutated clone changes the result.
                return doubled * 10 + working

        self._export_and_run(Model(), x, (x + 1) * 21)

    def test_clone_of_view_mutation_export_and_cpp_runtime(self) -> None:
        # One launch writing a clone of a view, on both backends: the copy
        # made for the write must read the view's elements, not the start
        # of its storage.
        kernel = self._kernel(
            _increment_and_double, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                working = x[1].clone()
                return kernel(working) + x[0]

        self._export_and_run(Model(), x, (x[1] + 1) * 2 + x[0])

    def test_direct_view_mutation_export_and_cpp_runtime(self) -> None:
        # A contiguous view written by the kernel: the copy made for the
        # write gathers through the view, so it reads row 1, not row 0.
        scale = self._kernel(
            _increment_and_double,
            helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                return scale(y[1]) + y[0] + y[1]

        self._export_and_run(Model(), x, (x[1] + 1) * 3 + x[0])

    def test_clone_and_copy_back_export_and_cpp_runtime(self) -> None:
        # Cloning a view, calling the kernel on the clone, then copying
        # it back also works, and the write reaches the original tensor.
        scale = self._kernel(
            _increment_and_double,
            helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                w = y[1].clone()
                out = scale(w)
                y[1].copy_(w)
                return out + y[0] + y[1]

        self._export_and_run(Model(), x, (x[1] + 1) * 3 + x[0])

    def test_rejects_strided_view_mutation(self) -> None:
        # A written view with gaps cannot be copied for the write without
        # changing its strides, so export rejects it.
        scale = self._kernel(
            _increment_and_double,
            helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(514, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                return scale(y[1::2])

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(
            NotImplementedError, "mutation clones that change strides"
        ):
            to_edge_transform_and_lower(
                program,
                partitioner=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
            )

    def test_transposed_view_mutation_export_and_cpp_runtime(self) -> None:
        # A dense but non-contiguous written view: the copy made for the
        # write must preserve its layout and read its elements.
        kernel = self._kernel(
            _increment_2d,
            helion.Config(block_sizes=[16, 64]),
            static_shapes=True,
        )
        x = torch.randn(128, 64, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = (x * 1).t()
                kernel(y)
                return y + y

        self._export_and_run(Model(), x, (x.t() + 1) * 2)

    def test_distinct_kernels_in_cond_branches_export_and_cpp_runtime(
        self,
    ) -> None:
        add = self._kernel(_add, helion.Config(block_sizes=[256]), static_shapes=True)
        add3 = self._kernel(_add3, helion.Config(block_sizes=[256]), static_shapes=True)
        x = torch.rand(257, device="cuda") + 0.5

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cond(
                    x.sum() > 0,
                    lambda x: add(x, x),
                    lambda x: add3(x, x, x),
                    (x,),
                )

        self._export_and_run(Model(), x, x * 2)
        self._export_and_run(Model(), -x, -3 * x)

    def test_functional_mutation_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _increment, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                result = x.clone()
                kernel(result)
                return result

        self._export_and_run(Model(), x, x + 1)

    def test_buffer_mutation_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _increment, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(257, device="cuda"))

            def forward(self, x):
                kernel(self.state)
                return self.state + x

        self._export_and_run(Model(), x, x + 1, expected_second=x + 2)

    def test_cond_on_view_operand_export_and_cpp_runtime(self) -> None:
        # The cond operand is a restored slice view; the branch kernel
        # must be specialized on its real row stride.
        kernel = self._kernel(
            _add_2d, helion.Config(block_sizes=[16, 64]), static_shapes=True
        )
        x = torch.rand(64, 256, device="cuda") + 0.5

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                return torch.cond(
                    x.sum() > 0,
                    lambda v: kernel(v, v),
                    lambda v: v.neg(),
                    (y[:, :128],),
                )

        self._export_and_run(Model(), x, x[:, :128] * 2)

    def test_cond_branch_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.rand(257, device="cuda") + 0.5

        class Model(torch.nn.Module):
            def forward(self, x):
                # No constant in the other branch: the CUDA backend's other
                # cond tests use the same shape, and a tensor constant inside
                # a branch trips an unrelated AOTI retrace error.
                return torch.cond(
                    x.sum() > 0, lambda x: kernel(x, x), lambda x: x.neg(), (x,)
                )

        self._export_and_run(Model(), x, x * 2)

    def test_conv1d_and_helion_export_and_cpp_runtime(self) -> None:
        # nn.Conv1d is in the CUDA backend's decomposition table, so the AOTI
        # backend retraces the program with run_decompositions after this
        # pass; the spliced launch must survive that retrace.
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(1, 4, 257, device="cuda")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1)

            def forward(self, x):
                y = self.conv(x)
                flat = y.reshape(-1)
                return kernel(flat, flat).view_as(y)

        model = Model().cuda()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # Measured routing can replace the conv with a Triton template,
        # leaving no decomposition-table op. run_decompositions retraces
        # the whole program even then, so force the backend's check on
        # instead of depending on which conv measures faster here.
        expected = model.conv(x) * 2
        with mock.patch.object(
            aoti_backend,
            "contains_any_call_fn_target_op",
            return_value=True,
        ) as contains:
            # Loose tolerances: the compiled conv accumulates differently
            # from eager.
            self._export_and_run(model, x, expected, rtol=1e-2, atol=1e-2)
        self.assertTrue(contains.called)

    def test_user_input_mutation_export_and_cpp_runtime(self) -> None:
        kernel = self._kernel(
            _increment, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                kernel(x)
                return x * 2

        # Outputs are the written-back mutated input, then the model output.
        self._export_and_run(Model(), x, (x + 1, (x + 1) * 2))

    def test_rejects_scalar_argument(self) -> None:
        kernel = self._kernel(
            _scale, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, 2.5)

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(
            NotImplementedError, "tensor-only arguments"
        ) as raised:
            LowerHelionPass()(program)
        self.assertTrue(any("_scale" in note for note in raised.exception.__notes__))

    def test_rejects_real_kernel_output_aliasing_input(self) -> None:
        kernel = self._kernel(
            _increment_and_return, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x.clone())

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(
            NotImplementedError, "does not yet support aliased outputs"
        ) as raised:
            to_edge_transform_and_lower(
                program,
                partitioner=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
            )
        notes = []
        error = raised.exception
        while error is not None:
            notes.extend(getattr(error, "__notes__", []))
            error = error.__cause__ or error.__context__
        self.assertTrue(any("_increment_and_return" in note for note in notes))

    def test_rejects_capture_missing_private_field(self) -> None:
        kernel = self._kernel(
            _increment, helion.Config(block_sizes=[256]), static_shapes=True
        )
        graph = Graph()
        value = graph.placeholder("x")
        with FakeTensorMode():
            value.meta["val"] = torch.empty((257,), device="cuda")
        node = graph.call_function(
            helion_kernel_wrapper_functional,
            kwargs={
                "kernel_idx": helion_kernel_side_table.add_kernel(kernel),
                "constant_args": {},
                "tensor_args": {"x": value},
                "output_spec": {
                    "mutated_inputs": ["x"],
                },
                "tensors_to_clone": ["x"],
            },
        )
        with self.assertRaisesRegex(RuntimeError, "lacks the expected field"):
            lower_helion._trace_helion_call(node)

    def test_user_input_mutation_updates_signature(self) -> None:
        kernel = self._kernel(
            _increment, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                kernel(x)
                return x * 2

        # The CUDA backend sees the functionalized Edge program.
        exported = torch.export.export(Model(), (x.clone(),), strict=True)
        program = to_edge(
            exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        ).exported_program()
        LowerHelionPass()(program)
        self.assertFalse(
            any("helion_kernel_wrapper" in str(n.target) for n in program.graph.nodes)
        )
        names = {n.name for n in program.graph.nodes}
        self.assertTrue(
            all(spec.arg.name in names for spec in program.graph_signature.output_specs)
        )
        actual = x.clone()
        torch.testing.assert_close(program.module()(actual), (x + 1) * 2)
        torch.testing.assert_close(actual, x + 1)

    def test_call_inside_cond_branch(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cond(
                    x.sum() > 0, lambda x: kernel(x, x), lambda x: x - 1, (x,)
                )

        program = torch.export.export(Model(), (x,), strict=True)
        LowerHelionPass()(program)
        program.validate()
        for module in program.graph_module.modules():
            self.assertFalse(
                any(
                    "helion_kernel_wrapper" in str(n.target) for n in module.graph.nodes
                )
            )
        for value in (x.abs(), -x.abs()):
            expected = value * 2 if value.sum() > 0 else value - 1
            torch.testing.assert_close(program.module()(value), expected)

    def test_cond_branch_sliced_operand_export_and_cpp_runtime(self) -> None:
        # The cond operand is a view one element into its storage. Inside the
        # branch it is a placeholder, but it is not a program input, so a
        # CuTe kernel must not be compiled as if it were fully aligned: the
        # exported program has to run, not fail its alignment check per call.
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.rand(257, device="cuda") + 0.5

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cond(
                    x.sum() > 0,
                    lambda v: kernel(v, v),
                    lambda v: v.neg(),
                    (x[1:],),
                )

        self._export_and_run(Model(), x, x[1:] * 2)

    def test_rejects_layout_change_of_exported_output(self) -> None:
        graph = Graph()
        original, replacement = graph.placeholder("original"), graph.placeholder(
            "replacement"
        )
        with FakeTensorMode():
            original.meta["val"] = torch.empty((4, 8), device="cuda")
            replacement.meta["val"] = torch.empty((8, 4), device="cuda").t()
        # A stride-only difference refreshes the recorded fake: the
        # replacement was traced from refreshed arguments.
        lower_helion._check_same_tensor_layout(original, replacement)
        self.assertEqual(original.meta["val"].stride(), (1, 4))
        with FakeTensorMode():
            replacement.meta["val"] = torch.empty((8, 4), device="cuda")
        with self.assertRaisesRegex(NotImplementedError, "recorded for the call as"):
            lower_helion._check_same_tensor_layout(original, replacement)

    def test_rejects_outputs_aliasing_inputs(self) -> None:
        traced = make_fx(lambda x: x[1:], tracing_mode="fake")(
            torch.randn(8, device="cuda")
        )
        with self.assertRaisesRegex(NotImplementedError, "alias inputs"):
            lower_helion._check_outputs_do_not_alias_inputs(traced)

    def test_rejects_capture_without_semantic_fingerprint(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        for node in program.graph.nodes:
            if "helion_kernel_wrapper" in str(node.target):
                spec = {
                    k: v
                    for k, v in node.kwargs["output_spec"].items()
                    if k != "_host_semantic_fingerprint"
                }
                node.kwargs = {**node.kwargs, "output_spec": spec}
        with self.assertRaisesRegex(RuntimeError, "host-semantic fingerprints"):
            LowerHelionPass()(program)

    def test_pass_does_not_tune_or_launch(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        for node in program.graph.nodes:
            if "helion_kernel_wrapper" in str(node.target):
                node.meta["stack_trace"] = "probe-stack-trace"
        with mock.patch.object(
            BoundKernel, "autotune", side_effect=AssertionError("unexpected tuning")
        ), mock.patch.object(
            BoundKernel,
            "__call__",
            side_effect=AssertionError("unexpected Helion launch"),
        ), mock.patch.object(
            JITFunction, "run", side_effect=AssertionError("unexpected launch")
        ), mock.patch.object(
            helion_cute, "_capture_launch", wraps=helion_cute._capture_launch
        ) as capture:
            LowerHelionPass()(program)
        # A CuTe launch never goes through JITFunction.run; capturing the
        # launch (instead of calling the entrypoint with Helion's launcher)
        # is what proves the pass did not launch it either.
        self.assertEqual(capture.call_count, 1 if self.BACKEND == "cute" else 0)
        program.validate()
        self.assertTrue(self._has_launch(program.graph))
        launches = [
            n
            for n in program.graph.nodes
            if self.LAUNCH_MARKER in str(n.target)
            or (n.args and self.LAUNCH_MARKER in str(n.args[0]))
        ]
        self.assertTrue(launches)
        self.assertEqual(
            [n.meta.get("stack_trace") for n in launches],
            ["probe-stack-trace"] * len(launches),
        )
        if self.BACKEND == "cute":
            launch = next(
                n for n in program.graph.nodes if self.LAUNCH_MARKER in str(n.target)
            )
            self.assertIn(torch.Tag.needs_exact_strides, launch.target.tags)
        self.assertFalse(
            any("helion_kernel_wrapper" in str(n.target) for n in program.graph.nodes)
        )
        torch.testing.assert_close(program.module()(x), x * 2)

    def test_rejects_dynamic_kernel(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=False
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(NotImplementedError, "static_shapes=True"):
            LowerHelionPass()(program)

    def test_rejects_mutation_clone_layout_change(self) -> None:
        kernel = self._kernel(
            _increment, helion.Config(block_sizes=[256]), static_shapes=True
        )
        graph = Graph()
        value = graph.placeholder("x")
        with FakeTensorMode():
            value.meta["val"] = torch.empty_strided((257,), (2,), device="cuda")
        node = graph.call_function(
            helion_kernel_wrapper_functional,
            kwargs={
                "kernel_idx": helion_kernel_side_table.add_kernel(kernel),
                "constant_args": {},
                "tensor_args": {"x": value},
                "output_spec": {
                    "leaf_specs": [],
                    "direct_aliases": {},
                    "mutated_inputs": ["x"],
                },
                "tensors_to_clone": ["x"],
            },
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "mutation clones that change strides",
        ):
            lower_helion._trace_helion_call(node)

    def test_rejects_changed_capture_semantics(self) -> None:
        kernel = self._kernel(
            _add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        for node in program.graph.nodes:
            if "helion_kernel_wrapper" in str(node.target):
                spec = dict(node.kwargs["output_spec"])
                spec["_host_semantic_fingerprint"] = "different capture"
                node.kwargs = {**node.kwargs, "output_spec": spec}
        with self.assertRaisesRegex(RuntimeError, "different host semantics"):
            LowerHelionPass()(program)

    def test_rejects_config_list(self) -> None:
        kernel = helion.kernel(
            backend=self.BACKEND,
            configs=[
                helion.Config(block_sizes=[128]),
                helion.Config(block_sizes=[256]),
            ],
            static_shapes=True,
        )(_add)
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(ValueError, "Select one Helion config offline"):
            LowerHelionPass()(program)

    def test_rejects_unselected_config(self) -> None:
        kernel = helion.kernel(backend=self.BACKEND, static_shapes=True)(_add)
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(ValueError, "Select one Helion config offline"):
            LowerHelionPass()(program)
