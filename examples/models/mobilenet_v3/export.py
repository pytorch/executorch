import logging

import torch
from torchvision import models

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)

# will refactor this in a separate file.
class MV3Model:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        logging.info("loading mobilenet_v3 model")
        mv3_small = models.mobilenet_v3_small(pretrained=True)
        logging.info("loaded mobilenet_v3 model")
        return mv3_small

    @staticmethod
    def get_example_inputs():
        tensor_size = (1, 3, 224, 224)
        return (torch.randn(tensor_size),)
