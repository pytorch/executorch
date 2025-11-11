import torch

from executorch.devtools import BundledProgram

from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from executorch.exir import to_edge_transform_and_lower
from torch.export import export


# Step 1: ExecuTorch Program Export
class SampleModel(torch.nn.Module):
    """An example model with multi-methods. Each method has multiple input and single output"""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("a", 3 * torch.ones(2, 2, dtype=torch.int32))
        self.register_buffer("b", 2 * torch.ones(2, 2, dtype=torch.int32))

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        z = x.clone()
        torch.mul(self.a, x, out=z)
        y = x.clone()
        torch.add(z, self.b, out=y)
        torch.add(y, q, out=y)
        return y


def main() -> None:
    """Sample code to generate bundled program and save it to file. It is the same as in https://pytorch.org/executorch/0.6/bundled-io.html#emit-example"""
    # Inference method name of SampleModel we want to bundle testcases to.
    # Notices that we do not need to bundle testcases for every inference methods.
    method_name = "forward"
    model = SampleModel()

    # Inputs for graph capture.
    capture_input = (
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
        (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
    )

    # Export method's FX Graph.
    method_graph = export(
        export(model, capture_input).module(),
        capture_input,
    )

    # Emit the traced method into ET Program.
    et_program = to_edge_transform_and_lower(method_graph).to_executorch()

    # Step 2: Construct MethodTestSuite for Each Method

    # Prepare the Test Inputs.

    # Number of input sets to be verified
    n_input = 10

    # Input sets to be verified.
    inputs = [
        # Each list below is a individual input set.
        # The number of inputs, dtype and size of each input follow Program's spec.
        [
            (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
            (torch.rand(2, 2) - 0.5).to(dtype=torch.int32),
        ]
        for _ in range(n_input)
    ]

    # Generate Test Suites
    method_test_suites = [
        MethodTestSuite(
            method_name=method_name,
            test_cases=[
                MethodTestCase(
                    inputs=input,
                    expected_outputs=(getattr(model, method_name)(*input),),
                )
                for input in inputs
            ],
        ),
    ]

    # Step 3: Generate BundledProgram
    bundled_program = BundledProgram(et_program, method_test_suites)

    # Step 4: Serialize BundledProgram to flatbuffer.
    serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )
    save_path = "bundled_program.bpte"
    with open(save_path, "wb") as f:
        f.write(serialized_bundled_program)


if __name__ == "__main__":
    main()
