# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.llava.export_llava import export_image_encoder
from executorch.examples.models.llava.model import LlavaModel
from torch import nn
from torchvision.transforms.v2 import functional as F
import torch

llava_model = LlavaModel(use_sdpa_with_kv_cache_op=True)

class ImagePreprocess(nn.Module):
    def __init__(self, mean, std, rescale_factor):
        super().__init__()
        self.image_mean = mean
        self.image_std = std
        self.rescale_factor = rescale_factor

    def forward(self, img):
        w = torch.max(torch.tensor([img.shape[1], img.shape[2]]))
        l_pad = (w - img.shape[2]) // 2
        r_pad = torch.floor_divide((w - img.shape[2]), 2)
        t_pad = (w - img.shape[1]) // 2
        b_pad = torch.floor_divide((w - img.shape[1]), 2)
        l = l_pad.item()
        r = r_pad.item()
        t = t_pad.item()
        b = b_pad.item()
        torch._check(l >= 0)
        torch._check(r >= 0)
        torch._check(t > 0)
        torch._check(t != 1)
        torch._check(b > 0)

        resized = F.pad(
            img,
            padding=(l_pad.item(), t_pad.item(), r_pad.item(), b_pad.item()),
            fill=tuple(int(x * 255) for x in self.image_mean),
        )
        # TODO: implement _upsample_bicubic_aa.out in portable kernel library.
        # here padded shape should be max(h, w) x max(h, w)
        # skipping resize for now due to missing _upsample_bicubic_aa kernel in portable
        # resized = resize(
        #     padded,
        #     size=[
        #         self.image_processor.crop_size["height"],
        #         self.image_processor.crop_size["width"],
        #     ],
        #     interpolation="bicubic",
        # )
        # torch._check(resized.size(1) == self.config.crop_size["height"])
        # torch._check(resized.size(2) == self.config.crop_size["width"])
        # print(resized.shape)
        # cropped = F.center_crop(img, output_size=[w, w])
        # print(cropped.shape)
        scaled = resized * self.rescale_factor
        # print(scaled)
        normed = F.normalize(
            scaled, self.image_mean, self.image_std
        )
        # print(normed)
        return normed.unsqueeze(0)
    
llava = llava_model.get_eager_model()
resized, = llava_model.get_example_inputs()
ip = ImagePreprocess(llava.image_processor.image_mean, llava.image_processor.image_std, llava.image_processor.rescale_factor)
pp = ip.forward(resized)
print(pp)

# export
ep = torch.export.export(ip, (resized,), dynamic_shapes=llava_model.get_dynamic_shapes(), strict=False)