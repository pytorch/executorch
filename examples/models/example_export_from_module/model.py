import torch

class myModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x

ModelUnderTest=myModel()
ModelInputs = (torch.ones(5),)
