from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch

op_params = [{'weights': True,  'bias': True,  'eps': 1.0     },
             {'weights': True,  'bias': True,  'eps': 0.00005 },
             {'weights': True,  'bias': True,  'eps': 0.5     },
             {'weights': True,  'bias': True,  'eps': 0.042   },
             {'weights': True,  'bias': False, 'eps': 1.0     },
             {'weights': True,  'bias': False, 'eps': 0.00005 },
             {'weights': True,  'bias': False, 'eps': 0.5     },
             {'weights': True,  'bias': False, 'eps': 0.042   },
             {'weights': False, 'bias': True,  'eps': 1.0     },
             {'weights': False, 'bias': True,  'eps': 0.00005 },
             {'weights': False, 'bias': True,  'eps': 0.5     },
             {'weights': False, 'bias': True,  'eps': 0.042   },
             {'weights': False, 'bias': False, 'eps': 1.0     },
             {'weights': False, 'bias': False, 'eps': 0.00005 },
             {'weights': False, 'bias': False, 'eps': 0.5     },
             {'weights': False, 'bias': False, 'eps': 0.042   },
             ]


class TestBatchNormOperator(BaseOpenvinoOpTest):

    def create_model(self, weights, bias, eps):

        class BatchNorm(torch.nn.Module):
            def __init__(self, weights=True, bias=True, eps=1e-05):
                super(BatchNorm, self).__init__()
                self.weight = torch.nn.Parameter(torch.randn(6)) if weights else None
                self.bias = torch.nn.Parameter(torch.randn(6)) if bias else None
                self.running_mean = torch.randn(6)
                self.running_var = torch.randn(6)
                self.eps = eps

            def forward(self, x):
                return torch.nn.functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, eps=self.eps, training=False)

        return BatchNorm(weights, bias, eps)


    def test_batch_norm(self):
        for params in op_params:
            with self.subTest(params=params):
                module = self.create_model(weights=params['weights'],
                                           bias=params['bias'],
                                           eps=params['eps'])

                sample_input = (torch.randn(20, 6, 10),)

                self.execute_layer_test(module, sample_input)
