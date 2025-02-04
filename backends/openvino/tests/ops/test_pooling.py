from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

d2_params = [{'kernel_size': [3, 3], 'stride': 1, 'padding': 0},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': 1},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [0, 1]},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [1, 0]},
             {'kernel_size': [3, 3], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': None, 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [], 'padding': 0},
             {'kernel_size': [8, 8], 'stride': [8, 4], 'padding': 1},
             ]

class TestPoolingOperator(BaseOpenvinoOpTest):

    def create_model(self, op_type, kernel_size, stride, padding, dilation=1, ceil_mode=True, count_include_pad=True, dtype=torch.float32):

        class MaxPoolingBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.ceil_mode = ceil_mode
                self.dtype = dtype

            def forward(self, x):
                pass

        class MaxPool2D(MaxPoolingBase):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x.to(self.dtype), self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode)

        class MaxPool2DIndices(MaxPoolingBase):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation,
                                                      self.ceil_mode, return_indices=True)
        
        ops = {
            "MaxPool2D": MaxPool2D,
            "MaxPool2DIndices": MaxPool2DIndices,
        }

        aten_pooling = ops[op_type]

        return aten_pooling()

    def test_pooling2d(self):
        for params in d2_params:
            with self.subTest(params=params):
                bias_shape = None
                if 'bias_shape' in params:
                    bias_shape = params['bias_shape']
                module = self.create_model(op_type='MaxPool2D',
                                           kernel_size=params['kernel_size'],
                                           stride=params['stride'],
                                           padding=params['padding'],
                                           dilation=1,
                                           ceil_mode=True,
                                           count_include_pad=True)
                sample_input = (torch.randn(1, 3, 15, 15),)
                self.execute_layer_test(module, sample_input)
