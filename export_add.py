import torch
from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.exir import to_edge
from torch.export import export


# Start with a PyTorch model that adds two input tensors (matrices)
class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # return triton_transpose_acc(x, y)
        return (x.cuda() + y.cuda()).cpu()


# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = export(
    Add(), (torch.ones(10, device="cpu"), torch.ones(10, device="cpu"))
)
# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)

edge_program = edge_program.to_backend(AotiPartitioner([]))

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("add.pte", "wb") as file:
    file.write(executorch_program.buffer)
