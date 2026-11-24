# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs the Helion lowering tests for kernels on Helion's CuTe backend."""

import importlib.util
import threading
import unittest

import executorch.backends.cuda.tests.test_lower_helion as helion_tests
import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes import helion_cute
from executorch.backends.cuda.passes.lower_helion import LowerHelionPass
from executorch.exir import to_edge_transform_and_lower

if (
    importlib.util.find_spec("helion") is None
    or importlib.util.find_spec("cutlass") is None
):
    raise unittest.SkipTest("Helion's CuTe backend is not installed")

# Outside any guard on purpose: a moved private module must fail, not skip.
from helion._compiler.cute.cutedsl_compat import check_cute_backend_requirements


def _cute_unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability() < (10, 0):
        return "Helion's CuTe backend targets Blackwell (SM100) GPUs"
    # On a GPU new enough for CuTe, a requirements failure is a broken DSL
    # install: fail loudly instead of skipping the whole suite.
    check_cute_backend_requirements()
    return None


_UNAVAILABLE_REASON = _cute_unavailable_reason()


@unittest.skipIf(_UNAVAILABLE_REASON is not None, _UNAVAILABLE_REASON or "")
class LowerHelionCuteTest(helion_tests.LowerHelionTest):
    BACKEND = "cute"
    LAUNCH_MARKER = "helion_cute"

    @unittest.skip("Helion's CuTe attention needs its own flash-attention config")
    def test_attention_export_and_cpp_runtime(self) -> None:
        pass

    def test_mixed_triton_and_cute_export_and_cpp_runtime(self) -> None:
        helion = helion_tests.helion
        triton_kernel = helion.kernel(
            backend="triton",
            config=helion.Config(block_sizes=[256]),
            static_shapes=True,
        )(helion_tests._add)
        cute_kernel = self._kernel(
            helion_tests._add, helion.Config(block_sizes=[256]), static_shapes=True
        )
        x = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                first = triton_kernel(x[0], x[1])
                return cute_kernel(first, x[1])

        self._export_and_run(Model(), x, x[0] + 2 * x[1])

    def test_chained_calls_share_one_launch(self) -> None:
        # The first call sees a user input (fully aligned), the second an
        # activation (element alignment): both must share the launch
        # exported at the weakest alignment, not split into two launches.
        kernel = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(257, device="cuda")
        w = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", w)

            def forward(self, x):
                return kernel(kernel(x, self.w), self.w)

        program = torch.export.export(Model(), (x,), strict=True)
        LowerHelionPass()(program)
        self.assertEqual(len(helion_cute._STATE.program_launches()), 1)
        self._export_and_run(Model(), x, x + w * 2)

    def test_rejects_two_distinct_cute_launches(self) -> None:
        add = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        scale = self._kernel(
            helion_tests._increment_and_double,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(2, 257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                working = x[1].clone()
                return add(x[0], scale(working))

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(
            NotImplementedError, "only one distinct CuTe launch"
        ):
            to_edge_transform_and_lower(
                program,
                partitioner=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
            )

    def test_view_buffer_export_and_cpp_runtime(self) -> None:
        # A buffer that is a view at an offset that is not 16-byte aligned
        # is compiled for element alignment and still runs correctly.
        kernel = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(257, device="cuda")
        storage = torch.randn(259, device="cuda")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", storage[1:258])

            def forward(self, x):
                return kernel(x, self.w)

        self._export_and_run(Model(), x, x + storage[1:258])

    def test_rejects_windows_target(self) -> None:
        kernel = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        with self.assertRaisesRegex(NotImplementedError, "Linux-only"):
            LowerHelionPass(target_platform="windows")(program)

    def test_eager_launch_rejects_unaligned_input(self) -> None:
        kernel = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        LowerHelionPass()(program)
        storage = torch.randn(258, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "layouts or alignments"):
            program.module()(storage[1:])

    def test_reexport_after_archive_deleted(self) -> None:
        kernel = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        # A size no other test uses, so the launch exports here first.
        x = torch.randn(259, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        with helion_cute._STATE.lock:
            before = set(helion_cute._STATE.exported)
        program = torch.export.export(Model(), (x,), strict=True)
        LowerHelionPass()(program)
        with helion_cute._STATE.lock:
            new = set(helion_cute._STATE.exported) - before
        self.assertEqual(len(new), 1)
        exported = helion_cute._STATE.exported[new.pop()]
        archive = exported.library_directory / f"lib{exported.library}.a"
        archive.unlink()
        program = torch.export.export(Model(), (x,), strict=True)
        LowerHelionPass()(program)
        rebuilt = (
            helion_cute._STATE.exported[exported.library].library_directory
            / f"lib{exported.library}.a"
        )
        self.assertTrue(rebuilt.is_file())
        torch.testing.assert_close(program.module()(x), x * 2)

    def test_export_visible_process_wide_links_per_program(self) -> None:
        # The hand-off contract other threads rely on: the exported op
        # and its shim declaration are process-wide, visible from any
        # thread's link context, while the fallback-kernel entry and the
        # archives link only into the program that recorded the launch.
        # (Concurrent full lowerings
        # still race inside torch's own global registrations, before any of
        # this state is read.)
        kernel = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        LowerHelionPass()(program)
        launches = list(helion_cute._STATE.program_launches().values())
        self.assertEqual(len(launches), 1)
        (exported,) = launches
        seen = []

        def observe() -> None:
            # No link context on this thread yet: if Inductor's config were
            # process-wide, the main thread's patch would be visible here.
            outside = torch._inductor.config.aot_inductor.custom_ops_to_c_shims.get(
                exported.op
            )
            with helion_cute.link_exported_launches():
                shims = torch._inductor.config.aot_inductor.custom_ops_to_c_shims
                seen.append(
                    (
                        outside,
                        shims.get(exported.op),
                        helion_cute.program_link_options(),
                    )
                )

        with helion_cute.link_exported_launches():
            fallbacks = helion_cute.exported_fallback_kernels()
            thread = threading.Thread(target=observe)
            thread.start()
            thread.join(timeout=60)
        self.assertFalse(thread.is_alive())
        ((outside_shim, shim, options),) = seen
        self.assertIsNone(outside_shim)
        self.assertIn(f"helion_cute::{exported.library}", fallbacks)
        self.assertEqual(shim, [exported.shim_declaration])
        # Archives are per program: another thread links none of them
        # unless its own program recorded the launch.
        self.assertEqual(options, ([], []))

    def test_prelowered_program_still_links_and_runs(self) -> None:
        # Lowering once by hand leaves helion_cute ops and no Helion calls;
        # the backend's own pass run must still link their archives.
        kernel = self._kernel(
            helion_tests._add,
            helion_tests.helion.Config(block_sizes=[256]),
            static_shapes=True,
        )
        x = torch.randn(257, device="cuda")

        class Model(torch.nn.Module):
            def forward(self, x):
                return kernel(x, x)

        program = torch.export.export(Model(), (x,), strict=True)
        LowerHelionPass()(program)
        edge = to_edge_transform_and_lower(
            program,
            partitioner=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
        )
        first, _second = helion_tests._run_twice(edge.to_executorch(), x)
        torch.testing.assert_close(torch.as_tensor(first[0], device="cuda"), x * 2)
        # The flag that keeps this program out of the PTX fallback set.
        self.assertTrue(helion_cute.last_program_had_launches())
