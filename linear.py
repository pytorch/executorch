import torch
from executorch import exir
from executorch.exir.print_program import print_program

class M(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(10, 10)

  def forward(self, x):
    return self.linear(x)

torch.manual_seed(0)
m = M()
inputs = (torch.randn(10, 10),)
print ("==== Inputs:", inputs)

print("==== exir.capture:")
# old api
# edge = exir.capture(m, inputs).to_edge()
# new api
edge = exir.capture(m, inputs, exir.CaptureConfig(enable_aot=True)).to_edge()

print("==== Printing edge.dump():")
edge.dump()

print("==== Running to_executorch() passes:")
exec_program = edge.to_executorch()

print("==== Printing program: ")
print(exec_program.program)

print("==== Print buffer: ")
print(exec_program.buffer)

filename = "linear_a9.pte"
print(f"=== write to {filename}")
with open(filename, "wb") as file:
  file.write(exec_program.buffer)
# emitter_output = program

# <serialize>
# this is when segments are extracted
