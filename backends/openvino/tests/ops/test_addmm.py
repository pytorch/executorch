# mypy: disable-error-code=import-not-found


from unittest import mock

import torch
from executorch.backends.openvino import preprocess
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)
from executorch.exir.backend.backend_details import CompileSpec
from openvino import Core, Type  # type: ignore[import-untyped]


class TestAddMMOperator(BaseOpenvinoOpTest):

    def create_model(self):
        class AddMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 1.0
                self.beta = 1.0

            def forward(self, x, y, z):
                # return torch.add(x, y)
                return torch.addmm(x, y, z, alpha=self.alpha, beta=self.beta)

        return AddMM()

    def test_addmm(self):
        module = self.create_model()
        input_x = torch.randn(4, 4, dtype=torch.float32)
        input_y = torch.randn(4, 4, dtype=torch.float32)
        input_z = torch.randn(4, 4, dtype=torch.float32)
        sample_input = (input_x, input_y, input_z)
        self.execute_layer_test(module, sample_input)

    def test_addmm_computes_in_float32(self):
        if not self.device.startswith("CPU"):
            self.skipTest("only CPUs compute in float32 by default")
        if Core().get_property("CPU", "INFERENCE_PRECISION_HINT") == Type.f32:
            self.skipTest("OpenVINO already computes in float32 on this CPU")
        # OpenVINO would otherwise compute this in bfloat16 on x86 CPUs with
        # bfloat16 units, or in float16 on ARM CPUs, far outside 1e-4.
        self.atol = self.rtol = 1e-4
        sample_input = tuple(torch.randn(256, 256) for _ in range(3))
        self.execute_layer_test(self.create_model(), sample_input)
        # On x86 only the runtime passes the precision on, so this fails if it
        # drops an explicit one.
        with self.assertRaisesRegex(AssertionError, "ref_output"):
            self.execute_layer_test(
                self.create_model(),
                sample_input,
                compile_specs=(CompileSpec("inference_precision", b"bf16"),),
            )

    def test_addmm_passes_the_precision_to_openvino(self):
        if not self.device.startswith("CPU"):
            self.skipTest("only CPUs compute in float32 by default")
        # Checks the lowering half too: on x86 the runtime half alone decides.
        self.atol = self.rtol = 0.05
        sample_input = tuple(torch.randn(4, 4) for _ in range(3))
        bf16 = (CompileSpec("inference_precision", b"bf16"),)
        for compile_specs, hint in (((), "f32"), (bf16, "bf16")):
            with self.subTest(hint=hint), mock.patch.object(
                preprocess, "compile_model", wraps=preprocess.compile_model
            ) as compile_model:
                self.execute_layer_test(
                    self.create_model(), sample_input, compile_specs=compile_specs
                )
                self.assertEqual(
                    compile_model.call_args.args[2], {"INFERENCE_PRECISION_HINT": hint}
                )

    def test_addmm_refuses_an_unknown_inference_precision_before_converting(self):
        sample_input = tuple(torch.randn(4, 4) for _ in range(3))
        with mock.patch.object(
            preprocess, "FrontEndManager", side_effect=AssertionError("converted")
        ), self.assertRaisesRegex(ValueError, "unsupported inference_precision 'f7'"):
            self.execute_layer_test(
                self.create_model(),
                sample_input,
                compile_specs=(CompileSpec("inference_precision", b"f7"),),
            )
