from transformers import MimiModel
import torch
import torch.nn as nn
from torch.export import export, export_for_training, ExportedProgram


mimi: nn.Module = MimiModel.from_pretrained("kyutai/mimi")
# print(mimi)

chunk = torch.ones(1,1,1024)
codes = mimi.encode(chunk)
print(codes)


class MimiEncode(nn.Module):
    def __init__(self, mimi: nn.Module):
        super().__init__()
        self.mimi_model = mimi

    def forward(self, x):
        return self.mimi_model.encode(x)

mimi_encode = MimiEncode(mimi)
out = mimi_encode(chunk)
# exported_encode = export_for_training(mimi_encode, (chunk,), strict=False).module()

class MimiDecode(nn.Module):
    def __init__(self, mimi: nn.Module):
        super().__init__()
        self.mimi_model = mimi

    def forward(self, x):
        return self.mimi_model.decode(x)

mimi_decode = MimiDecode(mimi)
decode_input = torch.ones(1, 32, 125, dtype=float)
exported_decode = export_for_training(mimi_decode, (decode_input,), strict=False).module()

