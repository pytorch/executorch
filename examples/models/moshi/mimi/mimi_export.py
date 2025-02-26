from transformers import MimiModel
import torch
import torch.nn as nn
from torch.export import export, export_for_training, ExportedProgram
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache

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

    def forward(self,
            input_values: torch.Tensor,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            return_dict: Optional[bool] = None,
            use_code: bool = False,
            requires_grad: bool = False,
        ):
        embeddings = input_values
        embeddings = self.mimi_model.upsample(embeddings)

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        decoder_outputs = self.mimi_model.decoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            past_key_values = decoder_outputs[1]
        embeddings = decoder_outputs[0].transpose(1, 2)
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        outputs = self.mimi_model.decoder(embeddings)
        return outputs


mimi_decode = MimiDecode(mimi)
decode_input = torch.ones(1, 512, 125, dtype=torch.float32)
decode_output = mimi_decode(decode_input)
exported_decode = export_for_training(mimi_decode, (decode_input,), strict=False).module()

