import logging

import torch
from torchvision import models

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)

# will refactor this in a separate file.
class MV2Model:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        logging.info("loading mobilenet_v2 model")
        mv2 = models.mobilenet_v2(pretrained=True)
        logging.info("loaded mobilenet_v2 model")
        return mv2

    @staticmethod
    def get_example_inputs():
        tensor_size = (1, 3, 224, 224)
        return (torch.randn(tensor_size),)
