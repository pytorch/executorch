# READ:
# ~/src/llava_diff to run in CPU
# pip install -I torch for newer torch version for fp16

import torch

import os

os.environ['HF_TOKEN'] = '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'
from torch import nn

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, load_images, process_images
# from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge

from llava.model.multimodal_encoder.builder import build_vision_tower

# model_path = "liuhaotian/llava-v1.6-vicuna-7b"
# only this one works
model_path = "liuhaotian/llava-v1.5-7b"

# not full model
# model_path = "liuhaotian/llava-pretrain-llama-2-7b-chat"
# torch.set_default_dtype(torch.float32)

# class ImageEncoder(nn.Module):
#     def __init__(self, model_path):
#         super().__init__()
#         tokenizer, self.model_, image_processor, context_len = load_pretrained_model(
#             model_path=model_path,
#             model_base=None,
#             model_name=get_model_name_from_path(model_path)
#         )
#
#     def forward(self, images_tensor):
#
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
# images = load_images([image_file])
# image_sizes = [x.size for x in images]
# images_tensor = process_images(
#     images,
#     image_processor,
#     model.config
# ).to(model.device, dtype=torch.float32)
#
# image_features = model.get_model().get_vision_tower()(images_tensor).float()


prompt = "What are the things I should be cautious about when I visit here?"
image_files = ['/Users/myuan/Downloads/pyturkeys.jpg']  # IMG_3997

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_files[0],
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, args.model_base, model_name, device_map="cpu", device="cpu"
)

# # uncomment this line for end to end eager mode run
# eval_model(args)

import torch
from torch import nn
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'
# import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torchvision.transforms._functional_tensor import resize
from torchvision.io import read_image
import torchvision
from helpers import plot

from torch.export import export
from torch._export import capture_pre_autograd_graph
from torchvision.transforms.v2 import functional as F
from torch.export import Dim
from dataclasses import dataclass

imagr = torchvision.io.read_image(image_files[0])
imagt = imagr.to(dtype=torch.float16)
print(imagt.shape)


# plot([imagt])

@dataclass
class PreprocessConfig:
    crop_size: dict
    image_mean: list[float]
    image_std: list[float]
    rescale_factor: float


class Preprocess(nn.Module):
    def __init__(self, config: PreprocessConfig):
        super().__init__()
        self.config = config

    def forward(self, img):
        w = max(img.shape[1], img.shape[2])
        padded = torchvision.transforms.v2.functional.center_crop(img, output_size=[w, w])
        resized = resize(padded, size=[self.config.crop_size['height'], self.config.crop_size['width']])
        torch._check(resized.size(1) == self.config.crop_size['height'])
        torch._check(resized.size(2) == self.config.crop_size['width'])
        scaled = resized * self.config.rescale_factor
        normed = torchvision.transforms.v2.functional.normalize(scaled, self.config.image_mean, self.config.image_std)
        return normed

class Encoder(nn.Module):
    def __init__(self, llava_model):
        super().__init__()
        self.model_ = llava_model

    def forward(self, images_tensor):
        features = self.model_.get_model().get_vision_tower()(images_tensor)
        features = self.model_.get_model().mm_projector(features)
        return features

class EncoderModel(nn.Module):
    def __init__(self, preprocessor, encoder):
        super().__init__()
        self.pre_processor_ = preprocessor
        self.encoder_ = encoder

    def forward(self, images_tensor):
        processed = self.pre_processor_(images_tensor)
        processed = torch.unsqueeze(processed, dim=0)
        features = self.encoder_(processed)
        return features


pre_config = PreprocessConfig(image_processor.crop_size, image_processor.image_mean, image_processor.image_std,
                              image_processor.rescale_factor)
preprocessor = Preprocess(pre_config)
preout = preprocessor(imagt)
# plot([imagt])
plot([imagr, preout])

print("input shape:")
print(imagt.shape)
print("prehout shape:")
print(preout.shape)

print(f"{type(imagt) = }, {imagt.dtype = }, {imagt.shape = }")

inputs = (imagt,)
# ex_program = capture_pre_autograd_graph(model, inputs)
# dynamic_shapes = [{0: torch.export.Dim('length', min=1, max=2048),1: torch.export.Dim('width', min=1, max=2048)}]
length = Dim('length', min=8, max=4094)
width = Dim('width', min=6, max=4092)  # 1346 <= 2*_width <= 2048
# width = 2*_width
dynamic_shapes = [{1: length, 2: width}]
# width = 2*_width
ex_program = torch.export.export(preprocessor, inputs, dynamic_shapes=dynamic_shapes, strict=False)
torch.export.save(ex_program, '/Users/myuan/temp/pre.pt2')
print(ex_program.graph)
out1 = ex_program.module()(imagt)
# torchvision.io.

encoder = Encoder(model)
eninput = torch.unsqueeze(preout, dim=0)
enout = encoder(eninput)
eninputs = (eninput,)
ex_encodermodel = torch.export.export(encoder, eninputs, strict=False)
torch.export.save(ex_encodermodel, '/Users/myuan/temp/encoder.pt2')

encodermodel = EncoderModel(preprocessor, encoder)
out_m = encodermodel(imagt)
ex_encodermodel = torch.export.export(encodermodel, inputs, dynamic_shapes=dynamic_shapes, strict=False)
torch.export.save(ex_encodermodel, '/Users/myuan/temp/test.pt2')

print(ex_encodermodel.graph)

images = load_images(image_files)
image_sizes = [x.size for x in images]
images_tensor = process_images(
    images,
    image_processor,
    model.config
).to(model.device, dtype=torch.float16)
print(imagt)
print(images_tensor[0])
plot([out, images_tensor[0]])


debugint = 1
