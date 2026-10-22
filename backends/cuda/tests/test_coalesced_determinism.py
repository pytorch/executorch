# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A coalesced program must give the same answer every time.

A program split across two backends hands buffers from one delegate to the next. If
the delegates do not agree on a stream, a delegate can read an output before the
work that writes it has run, and the program returns a different answer on each
call. Nothing else in the suite runs a program that crosses backends, so nothing
else would notice.

Needs the TensorRT delegate, so it skips when torch_tensorrt or its ExecuTorch
runtime is absent.
"""

import os
import tempfile
import unittest

import torch


def _build_coalesced_pte(outdir):
    """Export a model split across the TensorRT and CUDA backends.

    Returns the program path, the inputs, and the eager result for those inputs, so
    the caller can check the answer is right and not only self-consistent.
    """
    import torch_tensorrt
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import ExecutorchBackendConfig

    class Model(torch.nn.Module):
        def __init__(self, dim=256, depth=6):
            super().__init__()
            self.depth = depth
            # One buffer per step rather than one shared across all of them. A
            # single constant read by every island becomes the same placeholder
            # name several times over in the flattened graph, which is a separate
            # export defect and would fail here before reaching the run loop.
            for i in range(depth):
                self.register_buffer("mix%d" % i, torch.randn(dim))
            self.scales = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(dim)) for _ in range(depth)]
            )

        def forward(self, x):
            x = torch.relu(x)
            for i in range(self.depth):
                y = x * self.scales[i]
                y = torch.relu(y)
                y = y * getattr(self, "mix%d" % i)
                x = x + y
            return x

    torch.manual_seed(0)
    model = Model().eval().cuda()
    gen = torch.Generator(device="cuda").manual_seed(0)
    inputs = (torch.randn(8, 256, device="cuda", generator=gen),)

    with torch.inference_mode():
        eager = model(*inputs).cpu()
        exported = torch.export.export(model, inputs)
        # Withhold one operator from TensorRT so the graph has to split, which is
        # what puts a delegate boundary in the middle of the data flow.
        graph = torch_tensorrt.dynamo.compile(
            exported,
            inputs=list(inputs),
            enabled_precisions={torch.float32},
            min_block_size=1,
            truncate_double=True,
            torch_executed_ops={"torch.ops.aten.mul.Tensor"},
        )

    pte = os.path.join(outdir, "coalesced.pte")
    spec = CudaBackend.generate_method_name_compile_spec("forward")
    torch_tensorrt.save(
        graph,
        pte,
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        partitioners=[CudaPartitioner([spec])],
        backend_config=ExecutorchBackendConfig(),
    )

    return pte, inputs, eager


class TestCoalescedDeterminism(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        try:
            import torch_tensorrt
            import torch_tensorrt_executorch_runtime  # noqa: F401
        except (ImportError, OSError) as error:
            # OSError as well, since a shared library that fails to load raises that
            # rather than ImportError.
            self.skipTest("TensorRT delegate not installed: %s" % error)
        # Importing is not enough: saving in this format needs the C++ runtime, and
        # without it the export raises rather than this test skipping.
        if not getattr(
            torch_tensorrt.ENABLED_FEATURES, "torch_tensorrt_runtime", False
        ):
            self.skipTest("the TensorRT delegate is installed without its runtime")

    def test_repeated_execution_agrees(self):

        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as outdir:
            pte, inputs, eager = _build_coalesced_pte(outdir)
            # Read the program that will actually run, rather than the graph it
            # came from: the island count is fixed before the ExecuTorch lowering,
            # so it cannot say whether both backends ended up in the file.
            with open(pte, "rb") as f:
                written = f.read()
            self.assertIn(
                b"TensorRTBackend",
                written,
                "no TensorRT delegate in the saved program",
            )
            self.assertIn(
                b"CudaBackend", written, "no CUDA delegate in the saved program"
            )

            weights = [
                os.path.join(outdir, name)
                for name in sorted(os.listdir(outdir))
                if name.endswith(".ptd")
            ]
            # load_program takes one path, so a second file would be dropped and the
            # failure would look like a runtime bug.
            self.assertLessEqual(len(weights), 1, "expected at most one weights file")
            runtime = Runtime.get()
            program = (
                runtime.load_program(pte, data_path=weights[0])
                if weights
                else runtime.load_program(pte)
            )
            method = program.load_method("forward")

            host_inputs = [t.cpu() for t in inputs]
            first = method.execute(host_inputs)
            first = first[0] if isinstance(first, (list, tuple)) else first
            reference = first.clone()

            # Against eager with a tolerance, because a delegate that reads a stale
            # buffer the same way on every call is self-consistent and still wrong,
            # which the run-to-run check below cannot see.
            torch.testing.assert_close(reference, eager, rtol=1e-3, atol=1e-3)

            # Exact equality, not a tolerance: a delegate reading a buffer early
            # produces a different answer, not a slightly different one.
            for run in range(1, 100):
                out = method.execute(host_inputs)
                out = out[0] if isinstance(out, (list, tuple)) else out
                differing = int((out != reference).sum())
                largest = float((out - reference).abs().max())
                self.assertTrue(
                    torch.equal(out, reference),
                    "run %d disagreed with run 0 in %d element(s), largest "
                    "difference %g" % (run, differing, largest),
                )


if __name__ == "__main__":
    unittest.main()
