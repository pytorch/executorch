# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from model import LlavaModel
import torch
import torchvision
from torch.export import Dim

from executorch.exir import to_edge, EdgeCompileConfig
from executorch.examples.models.llama2.builder import LlamaEdgeManager, DType
from executorch.examples.models.llama2.export_llama_lib import WeightType
from executorch.extension.llm.export.partitioner_lib import get_xnnpack_partitioner
from executorch.examples.models.llama2.source_transformation.sdpa import replace_sdpa_with_custom_op
from executorch.examples.models.llama2.source_transformation.quantize import get_quant_weight_transform
from executorch.examples.models.llama2.export_llama_lib import build_args_parser, get_quantizer_and_quant_params
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.nn.attention import SDPBackend

def main():
    llava_model = LlavaModel()
    llava = llava_model.get_eager_model()

    llava = llava.to(torch.float32)  # overflow error with fp16

    import requests
    from PIL import Image
    image = Image.open(requests.get('https://llava-vl.github.io/static/images/view.jpg', stream=True).raw)
    # temp_file = "/Users/larryliu/Downloads/pyturkeys.jpg"
    temp_file = "./view.jpg"
    image.save(temp_file)
    imagr = torchvision.io.read_image(temp_file)

    class LlavaImageEncoder(torch.nn.Module):
        """ Takes images and prompts and encode them into embeddings. Result will be sent to the text model LlavaTextModel."""
        def __init__(self, llava):
            super().__init__()
            self.llava = llava

        def forward(self, image):
            return self.llava.image_embedding(image)
        

    class LlavaEdgeManager(LlamaEdgeManager):
        def __init__(self, **kwargs):
            if "dynamic_shapes" in kwargs:
                self.dynamic_shapes = kwargs.pop("dynamic_shapes")
            super().__init__(**kwargs)
        
        def _get_dynamic_shape(self) -> torch.Any:
            return self.dynamic_shapes
        
        def capture_pre_autograd_graph(self) -> "LlavaEdgeManager":
            dynamic_shape = self._get_dynamic_shape()
            # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
            # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
            with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
                self.export_program = torch.export.export(
                    self.model, self.example_inputs, dynamic_shapes=dynamic_shape, strict=False
                )
                self.pre_autograd_graph_module = self.export_program.module()
            return self
        
    height = Dim("height", min=1, max=336)
    width = Dim("width", min=28, max=336)
    dynamic_shapes = [{1: height, 2: width}]

    llava_image_encode = LlavaImageEncoder(llava)
    # input
    ratio = max(imagr.shape[1], imagr.shape[2]) / llava.image_processor.crop_size["height"]
    output_size = (int(imagr.shape[1] / ratio), int(imagr.shape[2] / ratio))
    resized = torchvision.transforms.Resize(size=output_size)(imagr)

    # quantizer
    linear_quantizer = XNNPACKQuantizer()
    operator_config_dynamic = get_symmetric_quantization_config(
        is_per_channel=True, is_dynamic=True
    )
    linear_quantizer.set_global(operator_config_dynamic)
    image_encoder_ep = LlavaEdgeManager(
        model=llava_image_encode,
        modelname="llava_image_encoder",
        weight_type=WeightType.LLAMA,
        dtype=DType.fp32,
        use_kv_cache=True,
        use_sdpa_with_kv_cache=True,
        example_inputs=(resized,),
        dynamic_shapes=dynamic_shapes,
    ).capture_pre_autograd_graph().pt2e_quantize([linear_quantizer]).export_to_edge()

    executorch_program = image_encoder_ep.to_backend([XnnpackQuantizedPartitioner()]).to_executorch()

if __name__ == "__main__":
    main()
