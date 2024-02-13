from executorch.util.export_edge_ir import export_to_edge
from executorch.util.export_mps import export_to_mps
import torch

m = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
example_inputs = (torch.rand(1, 3, 224, 224),)
# edge = export_to_edge(m, example_inputs)
export_to_mps(m, example_inputs)
