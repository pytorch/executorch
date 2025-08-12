import torch
from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.examples.models.mobilenet_v2 import MV2Model
from executorch.exir import to_edge
from torch.export import export
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights)
mv2 = mv2.eval()

model_inputs = (torch.randn(1, 3, 224, 224),)


# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = export(mv2, model_inputs)

# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)

edge_program = edge_program.to_backend(AotiPartitioner([]))

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("aoti_model.pte", "wb") as file:
    file.write(executorch_program.buffer)
