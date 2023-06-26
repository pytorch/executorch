import torch
from executorch.backends.xnnpack.test.test_xnnpack_utils import TestXNNPACK


class TestXNNPACKQuantizedModels(TestXNNPACK):
    def test_resnet18(self):
        import torchvision

        m = torchvision.models.resnet18().eval()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        self.quantize_and_test_model_with_quantizer(m, example_inputs)
