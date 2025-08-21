import sys

import torch
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class IndexModel(torch.nn.Module):
    def forward(self, x, n):
        return x[n]


class AddAllModel(torch.nn.Module):
    def forward(self, x, n):
        return x, n, x + n


if __name__ == "__main__":
    output_filepath = sys.argv[1] if len(sys.argv) > 1 else "test.pte"
    indexModel = IndexModel().eval()
    addAllModel = AddAllModel().eval()

    exported_index = export(indexModel, (torch.randn([3]), 1))
    exported_add_all = export(addAllModel, (torch.randn([2, 2]), 1))
    edge = to_edge_transform_and_lower(
        {
            "forward": exported_index,
            "index": exported_index,
            "add_all": exported_add_all,
        }
    )
    et = edge.to_executorch()
    with open(output_filepath, "wb") as file:
        file.write(et.buffer)
