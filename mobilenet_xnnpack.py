from executorch.util.export_edge_ir import export_to_edge
from executorch.util.export_xnnpack import export_to_xnnpack
import torch
import torchvision

m = torchvision.models.mobilenet_v2('DEFAULT')

example_inputs = (torch.rand(1, 3, 224, 224),)
exported_model = export_to_xnnpack(m, example_inputs, False)
