import torch
from torchtune.models.flamingo._component_builders import flamingo_vision_encoder
from executorch import exir
from executorch.examples.models.flamingo.preprocess import Preprocess, PreprocessConfig
from torchtune.models.clip._transforms import CLIPImageTransform
import numpy as np
import PIL
from torchvision.transforms.v2 import functional as F

transformer_config = {
    "tile_size" : 224,
    "patch_size" : 14,
    "num_heads" : 16,
    "embed_dim" : 1280,
    "num_layers_clip" : 32,
    "num_layers_adapter" : 8,
    "embed_dim_out" : 4096,
    "out_indices" : [3,7,15,23,30],
    "max_num_tiles" : 4,
    "in_channels" : 3,
}

image_transform = CLIPImageTransform(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        tile_size=224,
        possible_resolutions=None,
        max_num_tiles=4,
        resample="bilinear",
        resize_to_max_canvas=True,
    )

preprocess_config = PreprocessConfig(
    tile_size=224,#transformer_config["tile_size"],
    channels=3,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
    resample="bilinear",
    normalize=True,
)

preprocess_model = Preprocess(preprocess_config)
image = (np.random.rand(800,600,3) * 255).astype(np.uint8)
image_pil = PIL.Image.fromarray(image)
image_tensor = F.to_dtype(F.grayscale_to_rgb_image(F.to_image(image_pil)), scale=True)

target_size = torch.tensor([448, 336])
canvas_size = torch.tensor([448, 448])

preprocess_inputs = (image_tensor, target_size, canvas_size)
tune_image_transform = image_transform(image = image_pil)
print(f"tune_image_transform shape = {tune_image_transform['image'].shape}")
print(f"tune_image_transform = {tune_image_transform['aspect_ratio']}")

preprocess_outputs = preprocess_model(*preprocess_inputs)
print(f"preprocess_outputs.shape = {preprocess_outputs.shape}")
print(f"preprocess comparision {torch.allclose(tune_image_transform['image'], preprocess_outputs, atol=1e-06)}")

from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
with torch.no_grad():
    vision_transformer = flamingo_vision_encoder(**transformer_config).eval()
    aspect_ratio = torch.tensor([[2,2]]).reshape(1,1,2)
    print(vision_transformer)
    image = preprocess_outputs.reshape(
            (
                1,
                1,
                4,
                transformer_config["in_channels"],
                transformer_config["tile_size"],
                transformer_config["tile_size"],
            )
        )
    out = vision_transformer(image, aspect_ratio)
    ep = torch.export.export(vision_transformer, (image, aspect_ratio))
    outputs = ep.module()(image, aspect_ratio)
    edge = exir.to_edge(ep, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False))
    et = edge.to_executorch(
        config=exir.ExecutorchBackendConfig(
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )
    """
    module = _load_for_executorch_from_buffer(et.buffer)
    mod_outputs = module.forward((image, aspect_ratio))
    print(torch.allclose(outputs, mod_outputs[0], atol=1e-06))
    print(outputs.shape)
    print(mod_outputs[0])
    """
