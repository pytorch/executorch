from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

d2_params = [{'weights_shape': [3, 3, 2, 2], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [1, 1], 'groups': 1,
              'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 2, 2], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [
                 1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [
                 1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [
                 1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'bias_shape': [1], 'pads': [
                 1, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [1, 1], 'pads': [
                 1, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'bias_shape': [1], 'pads': [
                 3, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [1, 1], 'pads': [
                 3, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'bias_shape': [1], 'pads': [
                 1, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [1, 1], 'pads': [
                 0, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 1, 0], 'dilations': [1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 0, 1], 'dilations': [1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 1, 0], 'dilations': [2, 2], 'groups': 3, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 0, 0], 'dilations': [2, 2], 'groups': 3, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 1], 'bias_shape': [1], 'pads': [
                 1, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [2, 1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'bias_shape': [1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [2, 2], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 3, 1, 1], 'strides': [2, 1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'bias_shape': [1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'bias_shape': [1], 'pads': [
                 1, 1], 'dilations': [2, 2], 'groups': 1, 'output_padding': [1, 1], 'transposed': True},
             ]

class TestConvolutionOperator(BaseOpenvinoOpTest):

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias, transposed, output_padding=0,
                     bias_shape=None, underscore=False):

        bias_dim = 0

        class Convolution(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(weights_shape))
                self.bias_shape = bias_shape
                if self.bias_shape is None:
                    self.bias_shape = weights_shape[bias_dim]
                self.bias = torch.nn.Parameter(torch.randn(self.bias_shape)) if bias else None
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups
                self.transposed = transposed
                self.output_padding = output_padding
                if underscore:
                    self.forward = self.forward_
        
            def forward(self, x):
                return torch.convolution(
                    x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.transposed,
                    self.output_padding, self.groups
                )

            def forward_(self, x):
                return torch._convolution(
                    x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.transposed,
                    self.output_padding, self.groups, False, False, False, False
                )

        return Convolution()

    def test_convolution(self):
        bias_underscore_config = [(False, False), (True, False)]
        for bias, underscore in bias_underscore_config:
            for params in d2_params:
                with self.subTest(params=params, bias=bias, underscore=underscore):
                    bias_shape = None
                    if 'bias_shape' in params:
                        bias_shape = params['bias_shape']
                    module = self.create_model(weights_shape=params['weights_shape'],
                                               strides=params['strides'],
                                               pads=params['pads'],
                                               dilations=params['dilations'],
                                               groups=params['groups'],
                                               output_padding=params['output_padding'],
                                               transposed=params['transposed'],
                                               bias_shape=bias_shape,
                                               bias=bias,
                                               underscore=underscore)
                    sample_input = (torch.randn(1, 3, 10, 10),)
                    self.execute_layer_test(module, sample_input)
