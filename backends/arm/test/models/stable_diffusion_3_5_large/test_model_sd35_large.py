# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from executorch.backends.arm.test.models.stable_diffusion_3_5_large.test_configs_sd35_large import (
    get_tiny_sd35_large_t5_config,
    get_tiny_sd35_large_text_encoder_2_config,
    get_tiny_sd35_large_text_encoder_config,
    get_tiny_sd35_large_transformer_config,
    get_tiny_sd35_large_vae_config,
)
from executorch.examples.models.stable_diffusion_3_5_large import (
    model as sd35_large_model,
)
from transformers import CLIPTextModelWithProjection, T5EncoderModel


@pytest.mark.parametrize(
    ("clip_skip", "hidden_state_index"),
    (
        pytest.param(None, -2, id="default_clip_skip"),
        pytest.param(1, -3, id="clip_skip_1"),
    ),
)
def test_clip_text_encoder_wrapper_returns_selected_hidden_state_and_pooled_projection(
    clip_skip, hidden_state_index
):
    """Verify CLIP wrapper outputs."""
    config = get_tiny_sd35_large_text_encoder_config()
    config.num_hidden_layers = 3  # Set to 3 layers to test clip_skip=1 behavior
    text_encoder = CLIPTextModelWithProjection(config).to(dtype=config.dtype)
    text_encoder.eval()
    wrapper = sd35_large_model.SD3CLIPTextEncoderWrapper(
        text_encoder, clip_skip=clip_skip
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 7))

    with torch.no_grad():
        hidden_states, pooled_projection = wrapper(input_ids)
        expected = text_encoder(input_ids, output_hidden_states=True, return_dict=True)

    torch.testing.assert_close(
        hidden_states, expected.hidden_states[hidden_state_index]
    )
    torch.testing.assert_close(pooled_projection, expected[0])


def test_t5_text_encoder_wrapper_returns_last_hidden_state():
    """Verify T5 text encoder wrapper returns last hidden state."""
    config = get_tiny_sd35_large_t5_config()
    text_encoder = T5EncoderModel(config)
    text_encoder.eval()
    wrapper = sd35_large_model.SD3T5TextEncoderWrapper(text_encoder)
    input_ids = torch.randint(0, config.vocab_size, (2, 7))

    with torch.no_grad():
        hidden_states = wrapper(input_ids)
        expected = text_encoder(input_ids, return_dict=True)

    torch.testing.assert_close(hidden_states, expected.last_hidden_state)


def test_transformer_wrapper_returns_sample_tensor():
    """Verify transformer wrapper returns the sample tensor."""
    SD3Transformer2DModel = pytest.importorskip(
        "diffusers.models.transformers"
    ).SD3Transformer2DModel
    transformer = SD3Transformer2DModel(**get_tiny_sd35_large_transformer_config())
    transformer.eval()
    wrapper = sd35_large_model.SD3TransformerWrapper(transformer)
    batch_size = 2
    latents = torch.randn(batch_size, 4, 32, 32)
    timestep = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, 154, 16)
    pooled_projections = torch.randn(batch_size, 32)

    with torch.no_grad():
        sample = wrapper(
            latents,
            timestep,
            encoder_hidden_states,
            pooled_projections,
        )
        expected = transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=True,
        )

    torch.testing.assert_close(sample, expected.sample)


def test_vae_decoder_wrapper_scales_shifts_decodes_and_clamps():
    """Verify VAE decoder wrapper scales, shifts, decodes, and clamps."""
    AutoencoderKL = pytest.importorskip("diffusers.models.autoencoders").AutoencoderKL
    vae_config = get_tiny_sd35_large_vae_config()
    vae = AutoencoderKL(**vae_config)
    vae.eval()
    wrapper = sd35_large_model.SD3VAEDecoderWrapper(vae)
    latents = torch.randn(1, vae_config["latent_channels"], 4, 4)

    with torch.no_grad():
        image = wrapper(latents)
        expected_latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        expected = vae.decode(expected_latents, return_dict=True).sample
        # Normalize decoder output from [-1, 1] to image range [0, 1].
        expected = (expected / 2 + 0.5).clamp(0, 1)

    torch.testing.assert_close(image, expected)
    assert torch.all(image >= 0)
    assert torch.all(image <= 1)


@pytest.mark.parametrize(
    "getter_name",
    (
        "get_text_encoder_wrapper",
        "get_text_encoder_2_wrapper",
        "get_text_encoder_3_wrapper",
        "get_transformer_wrapper",
        "get_vae_decoder_wrapper",
    ),
)
def test_model_loader_getters_require_loaded_components(getter_name):
    """Verify model loader getters require loaded components."""
    loader = sd35_large_model.StableDiffusion3ModelLoader(dtype=torch.float32)

    with pytest.raises(ValueError, match="Models not loaded"):
        getattr(loader, getter_name)()


def test_model_loader_text_encoder_getters_wrap_loaded_components():
    """Verify model loader text encoder getters wrap loaded components."""
    loader = sd35_large_model.StableDiffusion3ModelLoader(dtype=torch.float32)
    loader.text_encoder = CLIPTextModelWithProjection(
        get_tiny_sd35_large_text_encoder_config()
    )
    loader.text_encoder_2 = CLIPTextModelWithProjection(
        get_tiny_sd35_large_text_encoder_2_config()
    )
    loader.text_encoder_3 = T5EncoderModel(get_tiny_sd35_large_t5_config())

    assert loader.get_text_encoder_wrapper().text_encoder is loader.text_encoder
    assert loader.get_text_encoder_2_wrapper().text_encoder is loader.text_encoder_2
    assert loader.get_text_encoder_3_wrapper().text_encoder is loader.text_encoder_3


def test_model_loader_transformer_getter_wraps_loaded_component():
    """Verify model loader transformer getter wraps loaded component."""
    SD3Transformer2DModel = pytest.importorskip(
        "diffusers.models.transformers"
    ).SD3Transformer2DModel
    loader = sd35_large_model.StableDiffusion3ModelLoader(dtype=torch.float32)
    loader.transformer = SD3Transformer2DModel(
        **get_tiny_sd35_large_transformer_config()
    )

    assert loader.get_transformer_wrapper().transformer is loader.transformer


def test_model_loader_vae_getter_wraps_loaded_component():
    """Verify model loader VAE getter wraps loaded component."""
    AutoencoderKL = pytest.importorskip("diffusers.models.autoencoders").AutoencoderKL
    loader = sd35_large_model.StableDiffusion3ModelLoader(dtype=torch.float32)
    loader.vae = AutoencoderKL(**get_tiny_sd35_large_vae_config())

    assert loader.get_vae_decoder_wrapper().vae is loader.vae


def _patch_model_loaders(monkeypatch):
    """Patch model loaders and return call records."""

    class FakeModel:
        def __init__(self):
            self.eval_called = False

        def to(self, dtype):
            return self

        def eval(self):
            self.eval_called = True
            return self

    calls = SimpleNamespace(
        tokenizer=[],
        text_encoder=[],
        t5=[],
        transformer=[],
        vae=[],
    )

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.tokenizer.append((model_id, kwargs))
            return SimpleNamespace(model_max_length=77)

    class FakeTextEncoder:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.text_encoder.append((model_id, kwargs))
            return FakeModel()

    class FakeT5:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.t5.append((model_id, kwargs))
            return FakeModel()

    class FakeTransformer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.transformer.append((model_id, kwargs))
            return FakeModel()

    class FakeVAE:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.vae.append((model_id, kwargs))
            return FakeModel()

    monkeypatch.setattr(sd35_large_model, "CLIPTokenizer", FakeTokenizer)
    monkeypatch.setattr(
        sd35_large_model, "CLIPTextModelWithProjection", FakeTextEncoder
    )
    monkeypatch.setattr(sd35_large_model, "T5EncoderModel", FakeT5)
    monkeypatch.setattr(sd35_large_model, "SD3Transformer2DModel", FakeTransformer)
    monkeypatch.setattr(sd35_large_model, "AutoencoderKL", FakeVAE)

    return calls


def test_load_models_uses_component_subfolders(monkeypatch):
    """Verify model loading uses the expected component subfolders."""
    calls = _patch_model_loaders(monkeypatch)

    loader = sd35_large_model.StableDiffusion3ModelLoader(
        model_id="test/sd3",
        dtype=torch.float32,
    )

    assert loader.load_models()
    assert calls.tokenizer == [
        ("test/sd3", {"subfolder": "tokenizer"}),
        ("test/sd3", {"subfolder": "tokenizer_2"}),
    ]
    assert calls.text_encoder == [
        ("test/sd3", {"subfolder": "text_encoder", "torch_dtype": torch.float32}),
        ("test/sd3", {"subfolder": "text_encoder_2", "torch_dtype": torch.float32}),
    ]
    assert calls.t5 == [
        ("test/sd3", {"subfolder": "text_encoder_3", "torch_dtype": torch.float32})
    ]
    assert calls.transformer == [
        ("test/sd3", {"subfolder": "transformer", "torch_dtype": torch.float32})
    ]
    assert calls.vae == [
        ("test/sd3", {"subfolder": "vae", "torch_dtype": torch.float32})
    ]
    assert loader.text_encoder.eval_called
    assert loader.text_encoder_2.eval_called
    assert loader.text_encoder_3.eval_called
    assert loader.transformer.eval_called
    assert loader.vae.eval_called


def test_load_models_loads_only_requested_component(monkeypatch):
    """Verify model loading can load only requested components."""
    calls = _patch_model_loaders(monkeypatch)

    loader = sd35_large_model.StableDiffusion3ModelLoader(
        model_id="test/sd3",
        dtype=torch.float32,
    )

    assert loader.load_models(
        [sd35_large_model.StableDiffusionComponent.TEXT_ENCODER_3]
    )
    assert calls.tokenizer == []
    assert calls.text_encoder == []
    assert calls.t5 == [
        ("test/sd3", {"subfolder": "text_encoder_3", "torch_dtype": torch.float32})
    ]
    assert calls.transformer == []
    assert calls.vae == []
    assert loader.text_encoder is None
    assert loader.text_encoder_2 is None
    assert loader.text_encoder_3 is not None
    assert loader.transformer is None
    assert loader.vae is None


@pytest.mark.parametrize(
    ("latent_size", "expected_latent_size"),
    (
        pytest.param(None, 32, id="default_latent_size"),
        pytest.param(16, 16, id="override_latent_size"),
    ),
)
def test_get_dummy_inputs_builds_expected_component_inputs(
    latent_size, expected_latent_size
):
    """Verify dummy inputs have expected component shapes."""
    loader = sd35_large_model.StableDiffusion3ModelLoader(dtype=torch.float32)
    loader.tokenizer = SimpleNamespace(model_max_length=77)
    loader.text_encoder = object()
    loader.text_encoder_2 = object()
    loader.text_encoder_3 = object()
    loader.transformer = SimpleNamespace(
        config=SimpleNamespace(
            in_channels=4,
            sample_size=32,
            joint_attention_dim=16,
            pooled_projection_dim=32,
        )
    )
    loader.vae = SimpleNamespace(config=SimpleNamespace(latent_channels=16))

    dummy_inputs = loader.get_dummy_inputs(
        max_sequence_length=256,
        latent_size=latent_size,
    )

    assert set(dummy_inputs) == {
        sd35_large_model.StableDiffusionComponent.TEXT_ENCODER,
        sd35_large_model.StableDiffusionComponent.TEXT_ENCODER_2,
        sd35_large_model.StableDiffusionComponent.TEXT_ENCODER_3,
        sd35_large_model.StableDiffusionComponent.TRANSFORMER,
        sd35_large_model.StableDiffusionComponent.VAE_DECODER,
    }
    assert dummy_inputs[sd35_large_model.StableDiffusionComponent.TEXT_ENCODER][
        0
    ].shape == (1, 77)
    assert (
        dummy_inputs[sd35_large_model.StableDiffusionComponent.TEXT_ENCODER][0].dtype
        == torch.long
    )
    assert dummy_inputs[sd35_large_model.StableDiffusionComponent.TEXT_ENCODER_2][
        0
    ].shape == (1, 77)
    assert dummy_inputs[sd35_large_model.StableDiffusionComponent.TEXT_ENCODER_3][
        0
    ].shape == (1, 256)

    transformer_inputs = dummy_inputs[
        sd35_large_model.StableDiffusionComponent.TRANSFORMER
    ]
    assert transformer_inputs[0].shape == (
        1,
        4,
        expected_latent_size,
        expected_latent_size,
    )
    assert transformer_inputs[0].dtype == torch.float32
    assert transformer_inputs[1].shape == (1,)
    assert transformer_inputs[1].dtype == torch.float32
    assert transformer_inputs[2].shape == (1, 333, 16)
    assert transformer_inputs[3].shape == (1, 32)

    assert dummy_inputs[sd35_large_model.StableDiffusionComponent.VAE_DECODER][
        0
    ].shape == (1, 16, expected_latent_size, expected_latent_size)
