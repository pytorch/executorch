from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

op_params = [{'axes': None,   'keep_dim': None,  'dtype': None,     },
             {'axes': None,   'keep_dim': None,  'dtype': "float64",},
             {'axes': None,   'keep_dim': None,  'dtype': "float32",},
             {'axes': None,   'keep_dim': None,  'dtype': "int32",  },
             {'axes': 0,      'keep_dim': False, 'dtype': None,     },
             {'axes': 0,      'keep_dim': False, 'dtype': None,     },
             ]

dtypes = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
    "uint8": torch.uint8
}

class TestMeanOperator(BaseOpenvinoOpTest):

    def create_model(self, axes, keep_dims, dtype):

        pt_dtype = dtypes.get(dtype)

        class Mean(torch.nn.Module):
            def __init__(self, axes=None, keep_dims=None, dtype=None):
                super(Mean, self).__init__()
                self.axes = axes
                self.keep_dims = keep_dims
                self.dtype = dtype

            def forward(self, x):
                if self.axes is None and self.keep_dims is None:
                    if self.dtype is None:
                        return torch.mean(x, dtype=self.dtype)
                    return torch.mean(x)
                if self.axes is not None and self.keep_dims is None:
                    if self.dtype is None:
                        return torch.mean(x, self.axes)
                    return torch.mean(x, self.axes, dtype=self.dtype)
                if self.dtype is None:
                    return torch.mean(x, self.axes, self.keep_dims)
                return torch.mean(x, self.axes, self.keep_dims, dtype=self.dtype)

        return Mean(axes, keep_dims, pt_dtype)


    def test_mean(self):
        for params in op_params:
            with self.subTest(params=params):
                module = self.create_model(axes=params['axes'],
                                           keep_dims=params['keep_dim'],
                                           dtype=params['dtype'])

                sample_input = (torch.randint(-10, 10, (1, 3, 224, 224)).to(dtype=torch.float32),)

                self.execute_layer_test(module, sample_input)
