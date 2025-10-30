# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from Hugging Face's diffusers library:
# https://github.com/huggingface/diffusers/blob/v0.33.1/tests/pipelines/stable_diffusion_3/test_pipeline_stable_diffusion_3.py
#
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers import CLIPTextConfig, T5Config


"""
This file defines test configs used to initialize Stable Diffusion module tests.
Module tests in the same directory will import these configs.

To stay aligned with the Stable Diffusion implementation in the HuggingFace Diffusers library,
the configs here are either directly copied from corresponding test files or exported from
pre-trained models used in the Diffusers library.

Licenses:
The test parameters are from Hugging Face's diffusers library and under the Apache 2.0 License,
while the remainder of the code is under the BSD-style license found in the LICENSE file in the
root directory of this source tree.
"""


# Source: https://github.com/huggingface/diffusers/blob/v0.33.1/tests/pipelines/stable_diffusion_3/test_pipeline_stable_diffusion_3.py#L56
CLIP_text_encoder_config = CLIPTextConfig(
    bos_token_id=0,
    eos_token_id=2,
    hidden_size=32,
    intermediate_size=37,
    layer_norm_eps=1e-05,
    num_attention_heads=4,
    num_hidden_layers=5,
    pad_token_id=1,
    vocab_size=1000,
    hidden_act="gelu",
    projection_dim=32,
)


# Source: https://github.com/huggingface/diffusers/blob/v0.33.1/tests/pipelines/stable_diffusion_3/test_pipeline_stable_diffusion_3.py#L76
# Exported from: T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5").config
T5_encoder_config = T5Config(
    bos_token_id=0,
    classifier_dropout=0.0,
    d_ff=37,
    d_kv=8,
    d_model=32,
    decoder_start_token_id=0,
    dense_act_fn="relu",
    dropout_rate=0.1,
    eos_token_id=1,
    feed_forward_proj="relu",
    gradient_checkpointing=False,
    initializer_factor=0.002,
    is_encoder_decoder=True,
    is_gated_act=False,
    layer_norm_epsilon=1e-06,
    model_type="t5",
    num_decoder_layers=5,
    num_heads=4,
    num_layers=5,
    pad_token_id=0,
    relative_attention_max_distance=128,
    relative_attention_num_buckets=8,
    transformers_version="4.47.1",
    vocab_size=1000,
)


# Source: https://github.com/huggingface/diffusers/blob/v0.33.1/tests/models/transformers/test_models_transformer_sd3.py#L142
SD3Transformer2DModel_init_dict = {
    "sample_size": 32,
    "patch_size": 1,
    "in_channels": 4,
    "num_layers": 4,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "caption_projection_dim": 32,
    "joint_attention_dim": 32,
    "pooled_projection_dim": 64,
    "out_channels": 4,
    "pos_embed_max_size": 96,
    "dual_attention_layers": (0,),
    "qk_norm": "rms_norm",
}


# Source: https://github.com/huggingface/diffusers/blob/v0.33.1/tests/pipelines/stable_diffusion_3/test_pipeline_stable_diffusion_3.py#L83
AutoencoderKL_config = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "block_out_channels": (4,),
    "layers_per_block": 1,
    "latent_channels": 4,
    "norm_num_groups": 1,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "shift_factor": 0.0609,
    "scaling_factor": 1.5035,
}
