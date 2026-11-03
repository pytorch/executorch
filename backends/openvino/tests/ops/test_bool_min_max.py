# mypy: disable-error-code=import-not-found

import torch
from executorch.backends.openvino.tests.ops.base_openvino_op_test import (
    BaseOpenvinoOpTest,
)


class TestBoolMinMax(BaseOpenvinoOpTest):
    """OpenVINO converts min, max, clamp and sign to ops that take no bool tensors."""

    def test_bool_ops_run_on_the_cpu(self):
        class BoolMinMax(torch.nn.Module):
            def forward(self, x):
                m = x > 0
                n = m.flip(1)
                return (
                    *m.min(1),
                    *m.max(1),
                    m.amin(1),
                    m.amax(1),
                    m.min(),
                    m.max(),
                    torch.minimum(m, n),
                    torch.maximum(m, n),
                    torch.sign(m),
                    m.clamp(min=0.5),
                )

        # One row all true and one all false, so min and max differ by row and
        # their indices must pick the first of the ties.
        x = torch.randn(3, 7)
        x[0] = x[0].abs() + 0.1
        x[1] = -x[1].abs() - 0.1
        self.execute_layer_test(BoolMinMax(), (x,), expected_partitions=2)

    def test_bool_min_of_a_buffer(self):
        class BufferMin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.tensor([1.0, -2.0, 3.0]))

            def forward(self, x):
                return x + (self.w > 0).min().float()

        # The comparison reads only the buffer, so its partition has no inputs.
        self.execute_layer_test(BufferMin(), (torch.randn(3),), expected_partitions=2)

    def test_maximum_of_bool_and_float_stays_with_openvino(self):
        class Maximum(torch.nn.Module):
            def forward(self, mask, x):
                return torch.maximum(mask, x)

        # Computed in float, which OpenVINO converts.
        x = torch.randn(2, 7)
        self.execute_layer_test(Maximum(), (x > 0, x))
