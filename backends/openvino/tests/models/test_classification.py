from executorch.backends.openvino.tests.ops.base_openvino_op_test import BaseOpenvinoOpTest
import torch
import timm
import torchvision.models as torchvision_models
from transformers import AutoModel

classifier_params = [
                     {'model': ['torchvision', 'resnet50', (1, 3, 224, 224)] },
                     {'model': ['torchvision', 'mobilenet_v2', (1, 3, 224, 224)] },
                    ]

# Function to load a model based on the selected suite
def load_model(suite: str, model_name: str):
    if suite == "timm":
        return timm.create_model(model_name, pretrained=True)
    elif suite == "torchvision":
        if not hasattr(torchvision_models, model_name):
            raise ValueError(f"Model {model_name} not found in torchvision.")
        return getattr(torchvision_models, model_name)(pretrained=True)
    elif suite == "huggingface":
        return AutoModel.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model suite: {suite}")

class TestClassifier(BaseOpenvinoOpTest):

    def test_classifier(self):
        for params in classifier_params:
            with self.subTest(params=params):
                module = load_model(params['model'][0], params['model'][1])

                sample_input = (torch.randn(params['model'][2]),)

                self.execute_layer_test(module, sample_input)
