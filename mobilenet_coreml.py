from executorch.util.export_edge_ir import export_to_edge
from executorch.util.export_coreml import export_to_coreml
import torch
import torchvision

m = torchvision.models.mobilenet_v2('DEFAULT')

example_inputs = (torch.rand(1, 3, 224, 224),)
coreml = export_to_coreml(m, example_inputs, "all")
