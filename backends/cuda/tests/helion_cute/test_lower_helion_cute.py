# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs the Helion lowering tests for kernels on Helion's CuTe backend."""

import threading
import unittest

import executorch.backends.cuda.tests.test_lower_helion as helion_tests
import torch
from executorch.backends.cuda.passes import helion_cute
from executorch.backends.cuda.passes.lower_helion import LowerHelionPass

try:
    from helion._compiler.cute.cutedsl_compat import check_cute_backend_requirements
except ImportError as error:
    raise unittest.SkipTest(
        f"Helion's CuTe backend is not installed: {error}"
    ) from error


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

    def test_export_visible_process_wide_but_links_per_thread(self) -> None:
        # The hand-off contract other threads rely on: the exported op, its
        # shim declaration and its fallback-kernel entry are visible process-
        # wide, while the archive link options stay on the exporting thread.
        # (Concurrent full lowerings race inside torch's own global
        # registrations, before any of this state is read.)
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
        with helion_cute._STATE.lock:
            exported = next(iter(helion_cute._STATE.exported.values()))
        seen = []

        def observe() -> None:
            with helion_cute.link_exported_launches():
                shims = torch._inductor.config.aot_inductor.custom_ops_to_c_shims
                seen.append(
                    (
                        helion_cute.exported_fallback_kernels(),
                        shims.get(exported.op),
                        helion_cute.program_link_options(),
                    )
                )

        thread = threading.Thread(target=observe)
        thread.start()
        thread.join(timeout=60)
        self.assertFalse(thread.is_alive())
        ((fallbacks, shim, options),) = seen
        self.assertIn(f"helion_cute::{exported.library}", fallbacks)
        self.assertEqual(shim, [exported.shim_declaration])
        self.assertEqual(options, ([], []))
        directories, libraries = helion_cute.program_link_options()
        self.assertIn(exported.library, libraries)
        self.assertIn(str(exported.library_directory), directories)
        helion_cute.begin_program()
