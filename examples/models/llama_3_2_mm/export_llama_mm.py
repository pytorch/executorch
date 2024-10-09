import math
import torch
import sys
import os
from torchtune.models.flamingo import flamingo_decoder, flamingo_vision_encoder, FlamingoTransform
from executorch import exir
from torchtune.models.convert_weights import get_mapped_key
from torchtune.data import Message
import PIL
from typing import Optional, Dict
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.data._prompt_templates import _TemplateType
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.passes import ToOutVarPass

in_channels = 3
tile_size = 560
max_num_tiles = 4

def llama3_2_vision_11b(decoder_trainable=False, encoder_trainable=True, fusion_trainable=True) -> DeepFusionModel:
    """ Llama 3.2 Vision 11B model

    Args:
        decoder_trainable (bool): Whether to make decoder params trainable. Default is False.
        encoder_trainable (bool): Whether to make encoder params trainable. Default is True.
        fusion_trainable (bool): Whether to make fusion params trainable. Default is True.

    Returns:
        DeepFusionModel: Instantiation of the Llama 3.2 Vision 11B model
    """
    encoder = flamingo_vision_encoder(
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1280,
        clip_num_layers=32,
        clip_hidden_states=[],
        decoder_embed_dim=4096,
        num_layers_projection=8,
        tile_size=560,
        max_num_tiles=4,
        in_channels=3,
    )
    decoder = flamingo_decoder(
        vocab_size=128_256,
        num_layers=32,
        fusion_interval=4,
        num_special_tokens=8,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        encoder_max_seq_len=64040,
        rope_base=500000.0,
        intermediate_dim=14336,
    )
    return DeepFusionModel(
        encoder=encoder,
        decoder=decoder,
        encoder_trainable=encoder_trainable,
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )
    
def flamingo_transform(path: str, max_seq_len: int = 8192, special_tokens_path: Optional[str] = None, prompt_template: Optional[_TemplateType] = None) -> FlamingoTransform:
    """
    Data Transforms (including Tokenizer) for Llama3 Vision.

    Args:
        path (str): path to the tokenizer
        max_seq_len (int): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file 
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.
    
    Returns:
        FlamingoTransform: Instantiation of the Llama3 tokenizer
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    template = _get_prompt_template(prompt_template) if prompt_template is not None else None
    return FlamingoTransform(
        path=path,
        special_tokens=special_tokens,
        tile_size=560,
        patch_size=14,
        max_num_tiles=4,
        max_seq_len=max_seq_len,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        prompt_template=template,
    )

_FROM_META = {
    "text_model.tok_embeddings.weight": "decoder.tok_embeddings.weight",
    "text_model.learnable_embedding.weight": "decoder.tok_embeddings.fusion_embedding.weight",
    "text_model.norm.weight": "decoder.norm.scale",
    "text_model.output.weight": "decoder.output.weight",
    "text_model.layers.{}.attention_norm.weight": "decoder.layers.{}.sa_norm.scale",
    "text_model.layers.{}.attention.wq.weight": "decoder.layers.{}.attn.q_proj.weight",
    "text_model.layers.{}.attention.wk.weight": "decoder.layers.{}.attn.k_proj.weight",
    "text_model.layers.{}.attention.wv.weight": "decoder.layers.{}.attn.v_proj.weight",
    "text_model.layers.{}.attention.wo.weight": "decoder.layers.{}.attn.output_proj.weight",
    "text_model.layers.{}.ffn_norm.weight": "decoder.layers.{}.mlp_norm.scale",
    "text_model.layers.{}.feed_forward.w1.weight": "decoder.layers.{}.mlp.w1.weight",
    "text_model.layers.{}.feed_forward.w3.weight": "decoder.layers.{}.mlp.w3.weight",
    "text_model.layers.{}.feed_forward.w2.weight": "decoder.layers.{}.mlp.w2.weight",
    "text_model.cross_attention_layers.{}.gate_attn": "decoder.layers.{}.fusion_layer.ca_scale.scale",
    "text_model.cross_attention_layers.{}.gate_ffwd": "decoder.layers.{}.fusion_layer.mlp_scale.scale",
    "text_model.cross_attention_layers.{}.attention_norm.weight": "decoder.layers.{}.fusion_layer.ca_norm.scale",
    "text_model.cross_attention_layers.{}.ffn_norm.weight": "decoder.layers.{}.fusion_layer.mlp_norm.scale",
    "text_model.cross_attention_layers.{}.attention.wq.weight": "decoder.layers.{}.fusion_layer.attn.q_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wk.weight": "decoder.layers.{}.fusion_layer.attn.k_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wv.weight": "decoder.layers.{}.fusion_layer.attn.v_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wo.weight": "decoder.layers.{}.fusion_layer.attn.output_proj.weight",
    "text_model.cross_attention_layers.{}.attention.q_norm.weight": "decoder.layers.{}.fusion_layer.attn.q_norm.scale",
    "text_model.cross_attention_layers.{}.attention.k_norm.weight": "decoder.layers.{}.fusion_layer.attn.k_norm.scale",
    "text_model.cross_attention_layers.{}.feed_forward.w1.weight": "decoder.layers.{}.fusion_layer.mlp.w1.weight",
    "text_model.cross_attention_layers.{}.feed_forward.w3.weight": "decoder.layers.{}.fusion_layer.mlp.w3.weight",
    "text_model.cross_attention_layers.{}.feed_forward.w2.weight": "decoder.layers.{}.fusion_layer.mlp.w2.weight",
    "vision_model.vision_encoder.positional_embedding": "encoder.clip.token_pos_embedding.local_token_positional_embedding",
    "vision_model.vision_encoder.gated_positional_embedding": "encoder.clip.token_pos_embedding.global_token_positional_embedding",
    "vision_model.vision_encoder.gated_positional_embedding_gate": "encoder.clip.token_pos_embedding.gate",
    "vision_model.vision_encoder.ln_pre.weight": "encoder.clip.ln_pre.weight",
    "vision_model.vision_encoder.ln_pre.bias": "encoder.clip.ln_pre.bias",
    "vision_model.vision_encoder.ln_post.weight": "encoder.clip.ln_post.weight",
    "vision_model.vision_encoder.ln_post.bias": "encoder.clip.ln_post.bias",
    "vision_model.vision_encoder.pre_tile_pos_embed.embedding": "encoder.clip.pre_tile_pos_embed.embedding",
    "vision_model.vision_encoder.pre_tile_pos_embed.gate": "encoder.clip.pre_tile_pos_embed.gate",
    "vision_model.vision_encoder.post_tile_pos_embed.embedding": "encoder.clip.post_tile_pos_embed.embedding",
    "vision_model.vision_encoder.post_tile_pos_embed.gate": "encoder.clip.post_tile_pos_embed.gate",
    "vision_model.vision_encoder.class_embedding": "encoder.clip.cls_token_embedding.weight",
    "vision_model.vision_encoder.conv1._linear.weight": "encoder.clip.conv.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wq.weight": "encoder.clip.layers.{}.attn.q_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wk.weight": "encoder.clip.layers.{}.attn.k_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wv.weight": "encoder.clip.layers.{}.attn.v_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wo.weight": "encoder.clip.layers.{}.attn.output_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_fc.weight": "encoder.clip.layers.{}.mlp.w1.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_fc.bias": "encoder.clip.layers.{}.mlp.w1.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_proj.weight": "encoder.clip.layers.{}.mlp.w2.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_proj.bias": "encoder.clip.layers.{}.mlp.w2.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_1.weight": "encoder.clip.layers.{}.sa_norm.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_1.bias": "encoder.clip.layers.{}.sa_norm.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_2.weight": "encoder.clip.layers.{}.mlp_norm.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_2.bias": "encoder.clip.layers.{}.mlp_norm.bias",
    "vision_model.vision_projection.weight": "encoder.projection.output.weight",
    "vision_model.vision_projection.bias": "encoder.projection.output.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wq.weight": "encoder.projection.layers.{}.attn.q_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wk.weight": "encoder.projection.layers.{}.attn.k_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wv.weight": "encoder.projection.layers.{}.attn.v_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wo.weight": "encoder.projection.layers.{}.attn.output_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.weight": "encoder.projection.layers.{}.mlp.w1.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.bias": "encoder.projection.layers.{}.mlp.w1.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.weight": "encoder.projection.layers.{}.mlp.w2.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.bias": "encoder.projection.layers.{}.mlp.w2.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_1.weight": "encoder.projection.layers.{}.sa_norm.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_1.bias": "encoder.projection.layers.{}.sa_norm.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_2.weight": "encoder.projection.layers.{}.mlp_norm.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_2.bias": "encoder.projection.layers.{}.mlp_norm.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.gate_attn": "encoder.projection.layers.{}.sa_scale.scale",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.gate_ffn": "encoder.projection.layers.{}.mlp_scale.scale",
}


def _layer_num(key: str):
    """Get layer number from key or return None"""
    layer_num = [int(k) for k in key.split(".") if k.isdigit()]
    if len(layer_num) > 1:
        raise ValueError("More than one number in key, ambiguous input")
    elif len(layer_num) == 1:
        return int(layer_num[0])
    else:
        return None


def flamingo_meta_to_tune(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convertor from Meta state dict to torchtune state dict. This handles:
    - Updateing the cross attention layer numbers
    """
    converted_state_dict = {}

    # Calculate fusion_interval: layer interval where cross attention layers are fused
    num_layers = max(_layer_num(k) for k in state_dict if "layers" in k) + 1
    num_fusion_layers = (
        max(_layer_num(k) for k in state_dict if "cross_attention_layers" in k) + 1
    )
    assert (
        num_layers % num_fusion_layers == 0
    ), "Conversion assumes cross attention is added at regular intervals"
    fusion_interval = num_layers // num_fusion_layers

    for key, value in state_dict.items():
        if key == "text_model.rope.freqs":
            continue
        new_key = get_mapped_key(key, _FROM_META)
        if "cross_attention_layers" in key:
            layer = int(key.split(".")[2])
            new_layer = (layer + 1) * fusion_interval - 1
            key_lst = new_key.split(".")
            key_lst[2] = str(new_layer)
            new_key = ".".join(key_lst)
            if "gate_ffwd" in key or "gate_attn" in key:
                value = value[:1]
        elif "conv1" in key:
            dim, flat_patch = value.shape
            patch_size = int(math.sqrt(flat_patch / 3))
            assert (
                3 * patch_size**2 == flat_patch
            ), "Conversion assumes 3 channel inputs and square patch size"
            value = value.reshape(dim, 3, patch_size, patch_size)
        converted_state_dict[new_key] = value
    return converted_state_dict

llama3_2_dir = str(sys.argv[1])
tokenizer_path = os.path.join(llama3_2_dir, "tokenizer.model")
checkpoint_path = os.path.join(llama3_2_dir, "consolidated.pth")
image_path = os.path.join(llama3_2_dir, "dog.jpg")

state_dict = torch.load(checkpoint_path)
state_dict = flamingo_meta_to_tune(state_dict)
model = llama3_2_vision_11b(decoder_trainable=False, encoder_trainable=False, fusion_trainable=False)
model.load_state_dict(state_dict)
transform = flamingo_transform(tokenizer_path)

images = [PIL.Image.open(image_path)]
messages = [
    Message(
        role="user",
        content=[
            {"type": "image", "content": images[0]},
            {"type": "text", "content": "What's in this image?"},
        ],
        eot=True,
    ),
    Message(role="assistant", content="")
]

with torch.no_grad():
    data = transform({"messages": messages}, inference=True)
    image_tokens = data["encoder_input"]["images"][0].reshape(
            (
                1,
                1,
                4,
                in_channels,
                tile_size,
                tile_size,
            )
        ).to(dtype=torch.float32)
    print(image_tokens[0].shape)
    aspect_ratio = data["encoder_input"]['aspect_ratio'][0].reshape(1,1,2).to(dtype=torch.int)
    dim = torch.export.Dim("num_tiles", min= 1,max=max_num_tiles)
    image_dynamic_dim = {
        0: 1,
        1: 1,
        2: dim,
        3: 3,
        4: tile_size,
        5: tile_size,
    }


    model.encoder = model.encoder.to(device="cpu", dtype=torch.float32)

    inputs_et = (image_tokens, aspect_ratio)
    out = model.encoder(*inputs_et)
    ep = torch.export.export(model.encoder, inputs_et, dynamic_shapes=(image_dynamic_dim,None))
    outputs = ep.module()(*inputs_et)
    assert torch.allclose(out.to(dtype=torch.float32, device="cpu"), outputs)

    edge = exir.program._program.to_edge_with_preserved_ops(ep, preserve_ops = [torch.ops.aten.scaled_dot_product_attention.default, torch.ops.aten.linear.default], compile_config=exir.EdgeCompileConfig(_check_ir_validity=False))
    outputs = edge.exported_program().module()(*inputs_et)

    et = edge.to_executorch(
        config=exir.ExecutorchBackendConfig(
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            to_out_var_pass = ToOutVarPass(ignore_to_out_var_failure=True)
        )
    )

    with open("/tmp/vision_model.pte", "wb") as file:
        file.write(et.buffer)

    print(f"Flatbuffer size = {len(et.buffer)/(1000*1000)}MB")
    print(f"Activations size = {et.executorch_program.execution_plan[0].non_const_buffer_sizes[1]/(1000*1000)}MB")

"""
with torch.no_grad():
    dim = torch.export.Dim("token_dim", min= 1,max=max_seq_len)
    dim_enc = torch.export.Dim("enc_dim", min= 1,max=2050)

    dyanmic_shapes = {
        "tokens": {0:1, 1:dim},
        "encoder_input": {0:1, 1:dim_enc, 2:4096},
        #"encoder_mask": {0:1, 1:dim, 2:dim_enc},
        "mask":None,
        "input_pos" : {0:dim},
    }

    tokens = torch.ones(1,64, dtype=torch.int)
    input_pos = torch.ones(64, dtype=torch.int)

    encoder_input = torch.ones(1, 2050, 4096)
    encoder_mask = torch.ones(1, 64, 2050)

    text_model.setup_caches(1, torch.float32)
    ep = torch.export.export(text_model, (tokens,),{"mask":None, "encoder_input":encoder_input, "input_pos":input_pos}, dynamic_shapes=dyanmic_shapes)
    edge = exir.to_edge(ep, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False))
    # Currently failing due to miscalculation of activations size. Fix pending.
    et = edge.to_executorch(
        config=exir.ExecutorchBackendConfig(
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )
"""
