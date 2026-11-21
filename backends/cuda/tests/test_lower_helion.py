# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import subprocess
import tempfile
import threading
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
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
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

    def test_dsl_runtime_directory_picks_newest_supported_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for flavor in ("cu11", "cu12", "cu13"):
                library = root / flavor / "lib" / "libcuda_dialect_runtime_static.a"
                library.parent.mkdir(parents=True)
                library.touch()
            spec = SimpleNamespace(submodule_search_locations=[str(root)])
            with mock.patch.object(
                helion_cute.importlib.util, "find_spec", return_value=spec
            ), mock.patch.object(torch.version, "cuda", "12.8"):
                self.assertEqual(
                    helion_cute._dsl_runtime_directory(), root / "cu12" / "lib"
                )
                with self.assertRaisesRegex(RuntimeError, "cannot find"):
                    with mock.patch.object(torch.version, "cuda", "10.2"):
                        helion_cute._dsl_runtime_directory()

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

    def test_fallback_kernels_cover_every_export(self) -> None:
        fake = helion_cute._ExportedLaunch(
            op=torch.ops.aten.add.Tensor,
            shim_declaration="AOTITorchError aoti_torch_cuda_q();",
            library="q",
            library_directory=Path("/tmp"),
        )
        with helion_cute._STATE.lock:
            helion_cute._STATE.exported["q_probe"] = fake
        try:
            helion_cute.begin_program()
            self.assertIn(
                "helion_cute::q_probe", helion_cute.exported_fallback_kernels()
            )
        finally:
            with helion_cute._STATE.lock:
                del helion_cute._STATE.exported["q_probe"]

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

    def test_collector_scopes_overlap_per_thread(self) -> None:
        original = CppWrapperCpu.generate_c_shim_extern_kernel_call
        barrier = threading.Barrier(2)
        results = {}

        def collect(name: str) -> None:
            missing: set[str] = set()
            with CudaBackend.collect_unsupported_fallback_kernels(missing):
                barrier.wait(timeout=60)
                aoti_backend.AotiBackend._record_missing_kernel(name)
            results[name] = missing

        threads = [
            threading.Thread(target=collect, args=(name,))
            for name in ("kernel_from_a", "kernel_from_b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        self.assertEqual(results["kernel_from_a"], {"kernel_from_a"})
        self.assertEqual(results["kernel_from_b"], {"kernel_from_b"})
        self.assertIs(CppWrapperCpu.generate_c_shim_extern_kernel_call, original)

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
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0),
        "the matmul consumer takes an unsupported TLX path on SM80",
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
                return kernel(y, y) + (y @ self.weight).sum()

        model = Model()
        y = x * 2
        expected = y * 2 + (y @ model.weight).sum()
        self._export_and_run(model, x, expected, rtol=1e-3, atol=1e-3)

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
        with self.assertRaisesRegex(NotImplementedError, "exported as"):
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
            NotImplementedError, "mutation clones that change strides"
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
