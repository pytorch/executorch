import torch


class myModelAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x


ModelUnderTest = myModelAdd()
ModelInputs = (torch.ones(5),)
