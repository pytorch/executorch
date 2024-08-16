import re
import torch
import PIL
from torchtune.models.flamingo._component_builders import flamingo_vision_encoder, flamingo_text_decoder
from torchtune.models.clip._transforms import CLIPImageTransform
from torchtune.modules.transforms import VisionCrossAttentionMask
from typing import Dict, Optional, List

def get_models():
    def flamingo_10b():
        return flamingo_text_decoder(
            vocab_size=128_256,
            num_layers=32,
            fusion_interval=4,
            num_special_tokens=8,
            num_heads=32,
            num_kv_heads=8,
            embed_dim=4096,
            max_seq_len=8192,
            intermediate_dim=14336,
            rope_base=500000.0,
        )
    def flamingo_10b_vision_encoder():
        return flamingo_vision_encoder(
            tile_size = 448,
            patch_size = 14,
            num_heads = 16,
            embed_dim = 1280,
            num_layers_clip = 32,
            num_layers_adapter = 8,
            embed_dim_out = 4096,
            out_indices = [3,7,15,23,30],
            max_num_tiles = 4,
            in_channels = 3,
            )

    text_model = flamingo_10b()
    vision_model = flamingo_10b_vision_encoder()
    return text_model, vision_model

def load_model_weights(text_model, vision_model, checkpoint_path:str):
    cp = torch.load(checkpoint_path)

    WEIGHTS_MAP: Dict[str, str] = {
        "tok_embeddings.weight": "tok_embeddings.embedding.weight",
        "norm.weight": "norm.scale",
        "learnable_embedding.weight": "tok_embeddings.fusion_embedding.weight",
        "layers.{}.gate_attn": "layers.{}.layer.attn_scale.scale",
        "layers.{}.gate_ffwd": "layers.{}.mlp_scale.scale",
        "layers.{}.attention.inner_attention.q_norm.weight": "layers.{}.attn.q_norm.scale",
        "layers.{}.attention.inner_attention.k_norm.weight": "layers.{}.attn.k_norm.scale",
        "layers.{}.attention_norm.weight": "layers.{}.attn_norm.scale",
        "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
        "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
        "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
        "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
        "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
        "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
        "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
        "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
        "cross_attention_layers.{}.gate_attn": "layers.{}.fusion_layer.attn_scale.scale",
        "cross_attention_layers.{}.gate_ffwd": "layers.{}.fusion_layer.mlp_scale.scale",
        "cross_attention_layers.{}.attention.inner_attention.q_norm.weight": "layers.{}.fusion_layer.attn.q_norm.scale",
        "cross_attention_layers.{}.attention.inner_attention.k_norm.weight": "layers.{}.fusion_layer.attn.k_norm.scale",
        "cross_attention_layers.{}.attention_norm.weight": "layers.{}.fusion_layer.attn_norm.scale",
        "cross_attention_layers.{}.attention.wq.weight": "layers.{}.fusion_layer.attn.q_proj.weight",
        "cross_attention_layers.{}.attention.wk.weight": "layers.{}.fusion_layer.attn.k_proj.weight",
        "cross_attention_layers.{}.attention.wv.weight": "layers.{}.fusion_layer.attn.v_proj.weight",
        "cross_attention_layers.{}.attention.wo.weight": "layers.{}.fusion_layer.attn.output_proj.weight",
        "cross_attention_layers.{}.ffn_norm.weight": "layers.{}.fusion_layer.mlp_norm.scale",
        "cross_attention_layers.{}.feed_forward.w1.weight": "layers.{}.fusion_layer.mlp.w1.weight",
        "cross_attention_layers.{}.feed_forward.w3.weight": "layers.{}.fusion_layer.mlp.w3.weight",
        "cross_attention_layers.{}.feed_forward.w2.weight": "layers.{}.fusion_layer.mlp.w2.weight",
        "vision_encoder.global_transformer.resblocks.{}.attn_scale.scale": "adapter.layers.{}.attn_scale.scale",
        "vision_encoder.global_transformer.resblocks.{}.gate_ffn": "adapter.layers.{}.mlp_scale.scale",
        "vision_encoder.global_transformer.resblocks.{}.attn.wq.weight": "adapter.layers.{}.attn.q_proj.weight",
        "vision_encoder.global_transformer.resblocks.{}.attn.wq.bias": "adapter.layers.{}.attn.q_proj.bias",
        "vision_encoder.global_transformer.resblocks.{}.attn.wk.weight": "adapter.layers.{}.attn.k_proj.weight",
        "vision_encoder.global_transformer.resblocks.{}.attn.wk.bias": "adapter.layers.{}.attn.k_proj.bias",
        "vision_encoder.global_transformer.resblocks.{}.attn.wv.weight": "adapter.layers.{}.attn.v_proj.weight",
        "vision_encoder.global_transformer.resblocks.{}.attn.wv.bias": "adapter.layers.{}.attn.v_proj.bias",
        "vision_encoder.global_transformer.resblocks.{}.attn.wo.weight": "adapter.layers.{}.attn.output_proj.weight",
        "vision_encoder.global_transformer.resblocks.{}.attn.wo.bias": "adapter.layers.{}.attn.output_proj.bias",
        "vision_encoder.global_transformer.resblocks.{}.ln_1.weight": "adapter.layers.{}.mlp_norm.weight",
        "vision_encoder.global_transformer.resblocks.{}.ln_1.bias": "adapter.layers.{}.mlp_norm.bias",
        "vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.weight": "adapter.layers.{}.mlp.layer1.weight",
        "vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.bias": "adapter.layers.{}.mlp.layer1.bias",
        "vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.weight": "adapter.layers.{}.mlp.layer2.weight",
        "vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.bias": "adapter.layers.{}.mlp.layer2.bias",
        "vision_encoder.global_transformer.resblocks.{}.ln_2.weight": "adapter.layers.{}.attn_norm.weight",
        "vision_encoder.global_transformer.resblocks.{}.ln_2.bias": "adapter.layers.{}.attn_norm.bias",
        "vision_encoder.global_transformer.resblocks.{}.gate_attn": "adapter.layers.{}.attn_scale.scale",
        "vision_encoder.transformer.resblocks.{}.attn_scale.scale": "vision_encoder.transformer_layers.{}.attn_scale.scale",
        "vision_encoder.transformer.resblocks.{}.gate_ffn": "vision_encoder.transformer_layers.{}.mlp_scale.scale",
        "vision_encoder.transformer.resblocks.{}.attn.wq.weight": "vision_encoder.transformer_layers.{}.attn.q_proj.weight",
        "vision_encoder.transformer.resblocks.{}.attn.wq.bias": "vision_encoder.transformer_layers.{}.attn.q_proj.bias",
        "vision_encoder.transformer.resblocks.{}.attn.wk.weight": "vision_encoder.transformer_layers.{}.attn.k_proj.weight",
        "vision_encoder.transformer.resblocks.{}.attn.wk.bias": "vision_encoder.transformer_layers.{}.attn.k_proj.bias",
        "vision_encoder.transformer.resblocks.{}.attn.wv.weight": "vision_encoder.transformer_layers.{}.attn.v_proj.weight",
        "vision_encoder.transformer.resblocks.{}.attn.wv.bias": "vision_encoder.transformer_layers.{}.attn.v_proj.bias",
        "vision_encoder.transformer.resblocks.{}.attn.wo.weight": "vision_encoder.transformer_layers.{}.attn.output_proj.weight",
        "vision_encoder.transformer.resblocks.{}.attn.wo.bias": "vision_encoder.transformer_layers.{}.attn.output_proj.bias",
        "vision_encoder.transformer.resblocks.{}.ln_1.weight": "vision_encoder.transformer_layers.{}.mlp_norm.weight",
        "vision_encoder.transformer.resblocks.{}.ln_1.bias": "vision_encoder.transformer_layers.{}.mlp_norm.bias",
        "vision_encoder.transformer.resblocks.{}.mlp.c_fc.weight": "vision_encoder.transformer_layers.{}.mlp.layer1.weight",
        "vision_encoder.transformer.resblocks.{}.mlp.c_fc.bias": "vision_encoder.transformer_layers.{}.mlp.layer1.bias",
        "vision_encoder.transformer.resblocks.{}.mlp.c_proj.weight": "vision_encoder.transformer_layers.{}.mlp.layer2.weight",
        "vision_encoder.transformer.resblocks.{}.mlp.c_proj.bias": "vision_encoder.transformer_layers.{}.mlp.layer2.bias",
        "vision_encoder.transformer.resblocks.{}.ln_2.weight": "vision_encoder.transformer_layers.{}.attn_norm.weight",
        "vision_encoder.transformer.resblocks.{}.ln_2.bias": "vision_encoder.transformer_layers.{}.attn_norm.bias",
        "vision_encoder.transformer.resblocks.{}.gate_attn": "vision_encoder.transformer_layers.{}.attn_scale.scale",
        "vision_encoder.class_embedding": "vision_encoder.cls_token_embedding.cls_embedding",
        "vision_encoder.conv1._linear.weight": "vision_encoder.conv._linear.weight",
        "vision_projection.weight": "adapter.projection.weight",
        "vision_projection.bias": "adapter.projection.bias",
        "vision_encoder.positional_embedding": "vision_encoder.token_pos_embedding.positional_embedding",
    }


    def cross_attn_layer(layer_num):
        return ((int(layer_num) - 3) % 4) == 0


    def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
        try:
            text_model = False
            if "text_model" in key:
                key = key.replace("text_model.", "")
                text_model = True
            if "vision_model" in key:
                key = key.replace("vision_model.", "")
            if "layers" in key or "resblocks" in key:
                # Replace layer number with "{}" to create key for lookup
                abstract_key = re.sub(r"(\.\d+)", ".{}", key)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = mapping_dict[abstract_key]
                if text_model:
                    if "cross_attention" in abstract_key:
                        layer_num = 4 * int(layer_num) + 3
                    elif cross_attn_layer(layer_num):
                        layer_num = layer_num + ".layer"
                new_key = new_key.format(layer_num)
            else:
                new_key = mapping_dict[key]
        except KeyError as e:
            new_key = key

        return new_key


    def meta_to_tune(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert a state dict from Meta's format to torchtune's format. State dicts
        from multiple checkpoint files should be consolidated into a single state dict
        before calling this function.

        Eg of Meta-format state dict can be found in the ``meta-llama/Llama-2-7b``
        repo in HF (https://huggingface.co/meta-llama/Llama-2-7b).

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.

        Returns:
            Dict[str, torch.Tensor]: State dict in torchtune's format.
        """
        converted_state_dict = {}
        for key, value in state_dict.items():
            if key not in ["rope.freqs"]:  # Skip loading the position embeddings
                new_key = get_mapped_key(key, WEIGHTS_MAP)
                converted_state_dict[new_key] = value

        return converted_state_dict

    cp = meta_to_tune(cp)

    text_model.load_state_dict(cp, strict = False)
    vision_model.load_state_dict(cp, strict = False)

def generate_image_tokens(image_path: str):
    image_transform = CLIPImageTransform(
        image_mean=None,
        image_std=None,
        tile_size=448,
        possible_resolutions=None,
        max_num_tiles=4,
        resample="bilinear",
        resize_to_max_canvas=True,
    )
    image = PIL.Image.open(image_path)
    output = image_transform(image = image)
    preprocess_image = output['image'] # [num_tiles, num_channels, tile_size, tile_size]
    aspect_ratio = output['aspect_ratio'].reshape(1,1,2)
    image = preprocess_image.reshape(
            (
                1,
                1,
                preprocess_image.size(0),
                preprocess_image.size(1),
                preprocess_image.size(2),
                preprocess_image.size(3),
            )
        )
    encoder_outputs = vision_model(image, aspect_ratio)
    return encoder_outputs, image

def multinomial_sample_one(probs: torch.Tensor) -> torch.Tensor:
    """Samples from a multinomial distribution."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = None
) -> torch.Tensor:
    """Generic sample from a probability distribution."""
    # scale the logits based on temperature
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # select the very last value from the top_k above as the pivot
        pivot = v.select(-1, -1).unsqueeze(-1)
        # set everything smaller than pivot value to inf since these
        # should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    # change logits into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_sample_one(probs)

#pyre-ignore
def generate_next_token(
    model: torch.nn.Module,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
) -> torch.Tensor:
    """Generates the next tokens."""
    _, _, _, num_image_tokens, image_token_dim = tuple(encoder_outputs.shape)
    encoder_outputs_reshaped = encoder_outputs.view(1, -1, image_token_dim)
    encoder_mask_new = mask['encoder_mask'][0].unsqueeze(0)
    # As new tokens are generated, the mask needs to be updated to reflect the new shape.
    if encoder_mask_new.shape[1] < x.shape[1]:
        while encoder_mask_new.shape[1] != x.shape[1]:
            last_row = encoder_mask_new[:, -1:, :]  # Select the last row along z dimension
            encoder_mask_new = torch.cat([encoder_mask_new, last_row], dim=1)  # Append the last row
    elif encoder_mask_new.shape[1] > x.shape[1]:
        encoder_mask_new = encoder_mask_new[:,:x.shape[1],:]
    logits = model(x, encoder_input = encoder_outputs_reshaped, encoder_mask=encoder_mask_new, input_pos=input_pos)[:, -1]
    return sample(logits, temperature, top_k)

@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> List[List[int]]:
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()

    # generate the first tokens conditioned on the prompt
    input_pos = torch.arange(0, model.max_seq_len, device=prompt.device)
    tokens = generate_next_token(
        model,
        input_pos=input_pos[:prompt_length],
        x=prompt,
        temperature=temperature,
        top_k=top_k,
    )
    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    curr_pos = prompt_length
    for _ in range(max_generated_tokens - 1):
        curr_input_pos = input_pos[: curr_pos + 1]
        tokens = generated_tokens.clone()
        tokens = generate_next_token(
            model,
            input_pos=curr_input_pos,
            x=tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1

    return generated_tokens.tolist()

checkopoint_path = None
image_path = None

text_model, vision_model = get_models()
if checkopoint_path is not None:
    load_model_weights(text_model, vision_model, checkpoint_path)
assert image_path is not None, "Image path needs to be set."
encoder_outputs, image = generate_image_tokens(image_path)

# Special token representing an image.
image_token_id = 128010
# Tokens generated for the text: "<|image|>\n What's in this image?"
tokens = [128006, 882, 128007, 271, 128010, 198, 3639, 596, 304, 420, 2217, 30, 128009]
# Generate a cross attention mask for the image tokens. More details on how it works is here:
# https://pytorch.org/torchtune/main/generated/torchtune.modules.transforms.VisionCrossAttentionMask.html
transform = VisionCrossAttentionMask(tile_size=448, patch_size=14, image_token_id=image_token_id)
mask = transform({"tokens":tokens, "images":[image.squeeze(0).squeeze(0)]})

prompt = torch.tensor(tokens, dtype=torch.int)
generated_tokens = generate(
    model=text_model,
    prompt=prompt,
    max_generated_tokens=50,
    temperature=0.7,
    top_k=2,
)
