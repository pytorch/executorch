import torch
from torchtune.models.flamingo._component_builders import flamingo_vision_encoder, flamingo_text_decoder
from executorch import exir
from executorch.examples.models.flamingo.preprocess import Preprocess, PreprocessConfig
from torchtune.models.clip._transforms import CLIPImageTransform
import numpy as np
import PIL
from torchvision.transforms.v2 import functional as F

vision_transformer_config = {
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

text_decoder_config = {
    "vocab_size" : 128_256,
    "num_layers" : 32,
    "fusion_interval" : 4,
    "num_special_tokens" : 8,
    "num_heads" : 32,
    "num_kv_heads" : 8,
    "embed_dim" : 4096,
    "max_seq_len" : 64,
    "rope_base" : 500000.0,
    "intermediate_dim" : 14336,
}

from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

def get_sample_preprocess_outputs():
    image = (np.random.rand(800,600,3) * 255).astype(np.uint8)
    image_pil = PIL.Image.fromarray(image)
    image_tensor = F.to_dtype(F.grayscale_to_rgb_image(F.to_image(image_pil)), scale=True)
    image_transform = CLIPImageTransform(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        tile_size=224,
        possible_resolutions=None,
        max_num_tiles=4,
        resample="bilinear",
        resize_to_max_canvas=True,
    )
    return image_transform(image = image_pil)
    preprocess_outputs = tune_image_transform['image']
    return preprocess_outputs

with torch.no_grad():
    vision_transformer = flamingo_vision_encoder(**vision_transformer_config).eval()
    
    preprocess_outputs = get_sample_preprocess_outputs()
    image = preprocess_outputs['image'].reshape(
            (
                1,
                1,
                4,
                vision_transformer_config["in_channels"],
                vision_transformer_config["tile_size"],
                vision_transformer_config["tile_size"],
            )
        )
    aspect_ratio = preprocess_outputs['aspect_ratio'].reshape(1,1,2)

    image_dynamic_dim = {
        0: 1,
        1: 1,
        # This should ideally be a dynamic dim with specs as follows:
        # torch.export.Dim("num_tiles", min=1, max=4).
        # We currently cannot specify this though because in some of the embedding
        # code we have a slice operation fails to export with dynamic dims. Once
        # that is fixed, we can update this to be a dynamic dim.
        2: 4,
        3: 3,
        4: 224,
        5: 224,
    }

    out = vision_transformer(image, aspect_ratio)
    ep = torch.export.export(vision_transformer, (image, aspect_ratio), dynamic_shapes=(image_dynamic_dim,None))
    outputs = ep.module()(image, aspect_ratio)
    edge = exir.to_edge(ep, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False))
    et = edge.to_executorch(
        config=exir.ExecutorchBackendConfig(
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )
    print(f"Flatbuffer size = {len(et.buffer)/(1000*1000)}MB")
    print(f"Activations size = {et.executorch_program.execution_plan[0].non_const_buffer_sizes[1]/(1000*1000)}MB")

with torch.no_grad():
    text_model = flamingo_text_decoder(**text_decoder_config).eval()


    dim = torch.export.Dim("token_dim", min= 1,max=text_decoder_config['max_seq_len'])
    dim_enc = torch.export.Dim("enc_dim", min= 1,max=2050)

    dyanmic_shapes = {
        "tokens": {0:1, 1:dim},
        "encoder_input": {0:1, 1:dim_enc, 2:4096},
        "encoder_mask": {0:1, 1:dim, 2:dim_enc},
        "mask":None,
        "input_pos" : {0:dim},
    }

    tokens = torch.ones(1,64, dtype=torch.int)
    input_pos = torch.ones(64, dtype=torch.int)

    encoder_input = torch.ones(1, 2050, 4096)
    encoder_mask = torch.ones(1, 64, 2050)

    ep = torch.export.export(text_model, (tokens,),{"mask":None, "encoder_input":encoder_input, "encoder_mask":encoder_mask, "input_pos":input_pos}, dynamic_shapes=dyanmic_shapes)
    edge = exir.to_edge(ep, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False))
    # Currently failing due to miscalculation of activations size. Fix pending.
    """
    et = edge.to_executorch(
        config=exir.ExecutorchBackendConfig(
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )
    """
