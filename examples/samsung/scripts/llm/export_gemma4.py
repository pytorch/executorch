#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Gemma 4 E2B to pte
"""

import argparse
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import torch
from executorch.backends.samsung.quantizer import EnnQuantizer, Precision
from executorch.backends.samsung.quantizer.custom_annotation import (
    custom_annotate_gemma4_text_decoder,
)
from executorch.backends.samsung.quantizer.observers import (
    MinMaxObserverWithFP16ZeroPoint,
)
from executorch.backends.samsung.quantizer.qconfig import _get_eps

from executorch.examples.samsung.scripts.llm.util.util import (
    free_memory,
    removing_gpu_node_in_graph,
    replace_rms_norm_with_native_rms_norm,
    seed_everything,
)
from executorch.examples.samsung.utils import save_position_embeddings, save_tensors

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve asset files relative to this file so the script can be run from any cwd.
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Assets are not checked in; they are downloaded from their original source on
# first use and cached in ASSET_DIR. Keep URLs pinned so calibration inputs
# stay reproducible.
ASSET_URLS = {
    "COCO_val2014_000000000730.jpg": (
        "https://images.cocodataset.org/val2014/COCO_val2014_000000000730.jpg"
    ),
    "COCO_val2014_000000002988.jpg": (
        "https://images.cocodataset.org/val2014/COCO_val2014_000000002988.jpg"
    ),
    "COCO_val2014_000000012966.jpg": (
        "https://images.cocodataset.org/val2014/COCO_val2014_000000012966.jpg"
    ),
    "COCO_val2014_000000013300.jpg": (
        "https://images.cocodataset.org/val2014/COCO_val2014_000000013300.jpg"
    ),
    "apps_sample-data_journal1.wav": (
        "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav"
    ),
}


def asset_path(filename: str) -> str:
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(path):
        url = ASSET_URLS.get(filename)
        if url is None:
            raise FileNotFoundError(
                f"Asset {filename} not found in {ASSET_DIR} and no download URL is known"
            )
        os.makedirs(ASSET_DIR, exist_ok=True)
        logger.info("Downloading %s from %s", path, url)
        urllib.request.urlretrieve(url, path)
    return path


GEMMA4_E2B_HF_MODEL = "google/gemma-4-E2B-it-qat-mobile-transformers"


def resolve_checkpoint(
    checkpoint_path: str | None, hf_model: str, revision: str | None
) -> str:
    """Return a local checkpoint directory, downloading hf_model if needed."""
    if checkpoint_path is not None:
        return checkpoint_path
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s from the Hugging Face Hub", hf_model)
    return snapshot_download(hf_model, revision=revision)


# Text-decoder graph names for the two autoregressive stages.
PREFILL_FORWARD = "text_decoder_prefill_forward"
DECODE_FORWARD = "text_decoder_decode_forward"
GRAPH_NAMES = [PREFILL_FORWARD, DECODE_FORWARD]


def get_image_input_list():
    messages_list = []
    messages_list.append(
        get_image_prompt(
            asset_path("COCO_val2014_000000000730.jpg"),
            "What are the colors of the bus in the image?",
        )
    )
    messages_list.append(
        get_image_prompt(
            asset_path("COCO_val2014_000000002988.jpg"),
            "What animal is lying on the chair?",
        )
    )
    messages_list.append(
        get_image_prompt(
            asset_path("COCO_val2014_000000012966.jpg"),
            "Describe the appearance of the dog.",
        )
    )
    messages_list.append(
        get_image_prompt(
            asset_path("COCO_val2014_000000013300.jpg"),
            "What kind of flowers are in the vase?",
        )
    )
    return messages_list


def get_text_input_list():
    messages_list = []
    messages_list.append(
        [
            {"role": "user", "content": "Write a short joke about saving RAM."},
        ]
    )
    messages_list.append(
        [
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    messages_list.append(
        [
            {"role": "user", "content": "Who are you?"},
        ]
    )
    messages_list.append(
        [
            {"role": "user", "content": "Please introduce the Gemma4 model."},
        ]
    )
    return messages_list


def _export_speech_transform(checkpoint_path: str):
    """Export speech transform model. Returns ExportedProgram."""
    from executorch.examples.samsung.scripts.llm.models.gemma4.speech_transform import (
        Gemma4SpeechTransformModel,
    )

    # Load audio config from HF checkpoint if available
    kwargs = {}
    try:
        # Audio feature extractor config is in processor_config.json (not config.json)
        processor_path = os.path.join(checkpoint_path, "processor_config.json")
        if os.path.exists(processor_path):
            with open(processor_path, "r") as f:
                processor_config = json.load(f)
            audio_fe = processor_config.get("feature_extractor", {})
            if audio_fe:
                field_map = {
                    "sampling_rate": "sample_rate",
                    "feature_size": "n_mels",
                    "min_frequency": "f_min",
                    "max_frequency": "f_max",
                    "mel_floor": "mel_floor",
                    "input_scale_factor": "input_scale_factor",
                    "frame_length": "frame_length",
                    "hop_length": "hop_length",
                }
                for hf_key, our_key in field_map.items():
                    if hf_key in audio_fe:
                        kwargs[our_key] = audio_fe[hf_key]
                logger.info(f"Speech transform config from checkpoint: {kwargs}")
    except Exception as e:
        logger.warning(
            f"Could not load audio config from checkpoint: {e}, using defaults"
        )

    model = Gemma4SpeechTransformModel(**kwargs)
    model.eval()

    sample_rate = kwargs.get("sample_rate", 16000)
    num_samples = sample_rate * 30  # 30s
    example = torch.randn(num_samples)
    min_samples = model.frame_length + model.hop_length + 1
    waveform_dim = torch.export.Dim("waveform_length", min=min_samples, max=960000)

    with torch.inference_mode():
        ep = torch.export.export(
            model, (example,), dynamic_shapes={"waveform": {0: waveform_dim}}
        )

    logger.info("Speech transform exported")
    del model
    free_memory()
    return ep


def _load_execu_models(
    checkpoint_path: str,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    max_patches: int = 1260,
    max_seq_len: int = 1024,
    variant: str = "e2b",
) -> tuple[dict, Any]:
    """Load HF Gemma4 once via Gemma4ForConditionalGeneration_ENN, extract sub-models.

    Replaces the previous approach of loading the full HF model separately for
    each encoder. The ENN model loads weights once, dequantizes
    QuantizedLinear/QuantizedEmbedding in-place, and produces three sub-models
    via init_execu_model().

    Returns:
        (execu_models, processor) where execu_models is a dict with keys:
        - "audio_encoder": AudioEncoderWithDynamicMask_ENN wrapper
        - "vision_encoder": VisionEncoderWithProjection_ENN wrapper
        - "text_decoder": Gemma4ForCausalLM (ExecuTorch native)
        - "full_model": Gemma4ForConditionalGeneration_ENN instance. Keep alive
          while audio_encoder/vision_encoder are in use (they reference
          full_model's submodules). text_decoder holds independent weight copies.
        and processor is the HuggingFace AutoProcessor for the checkpoint.
    """
    from executorch.examples.samsung.scripts.llm.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration_ENN,
    )
    from transformers import AutoConfig, AutoProcessor

    logger.info("Loading HF Gemma4ForConditionalGeneration_ENN...")
    config = AutoConfig.from_pretrained(checkpoint_path)

    processor = AutoProcessor.from_pretrained(checkpoint_path)

    pooling_kernel_size = config.vision_config.pooling_kernel_size
    SOFT_TOKENS_PER_IMAGE = max_patches // (pooling_kernel_size**2)

    processor.image_processor.max_soft_tokens = SOFT_TOKENS_PER_IMAGE
    # Nominal placeholders-per-image budgets.
    processor.image_processor.image_seq_length = SOFT_TOKENS_PER_IMAGE
    processor.image_seq_length = SOFT_TOKENS_PER_IMAGE
    # Model-side vision soft-token budget.
    config.vision_config.default_output_length = SOFT_TOKENS_PER_IMAGE
    config.vision_soft_tokens_per_image = SOFT_TOKENS_PER_IMAGE

    config.max_token_len = max_seq_len
    config.variant = variant

    model = Gemma4ForConditionalGeneration_ENN.from_pretrained(
        checkpoint_path,
        config=config,
        torch_dtype=dtype,
        device_map=device,
    )
    model.generation_config.do_sample = False
    logger.info("Initializing execu sub-models (audio/vision/text_decoder)...")
    model.model.init_execu_model(dtype, device, checkpoint_path)
    replace_rms_norm_with_native_rms_norm(model)

    return {
        "audio_encoder": model.model.audio_encoder,
        "vision_encoder": model.model.vision_encoder,
        "text_decoder": model.model.text_decoder,
        "full_model": model,
    }, processor


def _export_audio_encoder(
    checkpoint_path: str,
    model: torch.nn.Module,
):
    """Export audio encoder from a pre-loaded model. Returns ExportedProgram.

    The model is expected to come from _load_execu_models() (an
    AudioEncoderWithDynamicMask_ENN wrapper). Dynamic mask patching is
    already applied inside Gemma4Model_ENN.get_audio_encoder().
    """
    from executorch.examples.samsung.scripts.llm.models.gemma4.modeling_gemma4 import (
        _compute_mel_frame_count,
    )

    valid_frames = _compute_mel_frame_count(checkpoint_path)
    logger.info(f"  Input frames={valid_frames}")

    example_features = torch.randn(1, valid_frames, 128)
    example_mask = torch.ones(1, valid_frames, dtype=torch.bool)

    with torch.inference_mode():
        ep = torch.export.export(
            model,
            (example_features, example_mask),
        )

    logger.info("Audio encoder exported")
    free_memory()
    return ep


def _export_vision_encoder(
    model: torch.nn.Module,
    full_model: torch.nn.Module,
    max_patches: int = 1260,
    artifact_dir: str | None = None,
):
    """Export vision encoder from a pre-loaded model. Returns ExportedProgram.

    The model is expected to come from _load_execu_models() (a
    VisionEncoderWithProjection_ENN wrapper). full_model
    (Gemma4ForConditionalGeneration_ENN) provides _prepare_vision_input() for
    example inputs and dynamic shapes.

    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    (example_pixels, example_positions), _ = full_model._prepare_vision_input(
        device, dtype, max_patches
    )

    if artifact_dir is not None:
        from executorch.examples.models.gemma4.image_utils import preprocess_image

        image_path = asset_path("COCO_val2014_000000012966.jpg")
        pooling_kernel_size = full_model.config.vision_config.pooling_kernel_size
        pixel_values, pixel_position_ids, _ = preprocess_image(
            image_path, max_soft_tokens=max_patches // (pooling_kernel_size**2)
        )
        with torch.inference_mode():
            example_outputs = model.forward(pixel_values, pixel_position_ids)
        vision_dir = os.path.join(artifact_dir, "vision_encoder_in_out")
        os.makedirs(vision_dir, exist_ok=True)
        save_tensors((pixel_values, pixel_position_ids), "float_in", vision_dir)
        save_tensors(example_outputs, "float_out", vision_dir)

    with torch.inference_mode():
        ep = torch.export.export(
            model,
            (example_pixels, example_positions),
        )

    logger.info("Vision encoder exported")
    free_memory()
    return ep


def _export_text_decoder_graph(
    model: torch.nn.Module,
    example_inputs: dict,
    dynamic_shapes=None,
):
    """Export the text decoder graph on given example inputs."""
    with torch.inference_mode():
        return torch.export.export(
            model,
            (example_inputs["input_ids"],),
            kwargs={
                "inputs_embeds": example_inputs["inputs_embeds"],
                "mm_mask": example_inputs["mm_mask"],
                "attention_mask": example_inputs["attention_mask"],
                "position_embeddings_global": example_inputs[
                    "position_embeddings_global"
                ],
                "position_embeddings_local": example_inputs[
                    "position_embeddings_local"
                ],
                "k_cache": example_inputs["k_cache"],
                "v_cache": example_inputs["v_cache"],
            },
            dynamic_shapes=dynamic_shapes,
        )


def _export_text_decoder(
    model: torch.nn.Module,
    full_model: torch.nn.Module,
    prefill_len: int = 128,
    artifact_dir: str | None = None,
):
    """Export text decoder from a pre-loaded model into two graphs.

    Returns (programs, metadata) where programs maps each graph in GRAPH_NAMES
    (PREFILL_FORWARD, DECODE_FORWARD) to an ExportedProgram. The prefill graph
    is exported with prefill_len tokens, the decode graph with a single token.

    The model is expected to come from _load_execu_models() (a Gemma4ForCausalLM
    produced by init_execu_model). full_model (Gemma4ForConditionalGeneration_ENN)
    provides _prepare_text_input() for example inputs and dynamic shapes.

    """
    # Example inputs and dynamic shapes from full_model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    programs = {}
    for graph_name in GRAPH_NAMES:
        num_tokens = prefill_len if graph_name == PREFILL_FORWARD else 1
        logger.info(f"  Exporting {graph_name} (num_tokens={num_tokens})")

        example_inputs, _ = full_model._prepare_text_input(
            device, dtype, num_tokens=num_tokens
        )

        if artifact_dir is not None:
            with torch.inference_mode():
                example_outputs = model.forward(**example_inputs)
            graph_dir = os.path.join(artifact_dir, f"{graph_name}_in_out")
            os.makedirs(graph_dir, exist_ok=True)
            save_tensors(example_inputs, "float_in", graph_dir)
            save_tensors(example_outputs, "float_out", graph_dir)

        ep = _export_text_decoder_graph(model, example_inputs)
        programs[graph_name] = ep

    logger.info("Text decoder exported")
    free_memory()
    return programs


def get_tokenize_input(messages, processor, device, dtype):
    inputs = (
        processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        .to(device)
        .to(dtype)
    )
    input_len = inputs["input_ids"].shape[-1]
    return inputs, input_len


def get_text_input():
    messages = [
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]
    return messages


def get_image_prompt(img_path, text):
    from PIL import Image

    img = Image.open(img_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }
    ]
    return messages


def get_image_input():
    messages = get_image_prompt(
        asset_path("COCO_val2014_000000012966.jpg"),
        "Describe the appearance of the dog.",
    )
    return messages


def get_audio_input():
    import librosa

    audio_array, sampling_rate = librosa.load(
        asset_path("apps_sample-data_journal1.wav"), sr=16000, mono=True
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array, "sampling_rate": sampling_rate},
                {
                    "type": "text",
                    "text": "Transcribe the following speech segment in its original.",
                },
            ],
        }
    ]
    return messages


def _make_enn_quantizer(precision_name: str, act_observer_cls=None) -> EnnQuantizer:
    """Build an EnnQuantizer configured for a Precision enum member name.

    The gemma4 sub-models (text decoder, vision encoder, audio encoder) pass
    _get_eps so eps is resolved per dtype (int16 -> 0.0001/65535,
    int8 -> DEFAULT_EPS); other models default to DEFAULT_EPS.
    """
    quantizer = EnnQuantizer()
    if act_observer_cls is None:
        quantizer.setup_quant_params(
            getattr(Precision, precision_name), True, False, eps=_get_eps
        )
    else:
        quantizer.setup_quant_params(
            getattr(Precision, precision_name),
            True,
            False,
            act_observer_cls=act_observer_cls,
            eps=_get_eps,
        )
    return quantizer


def _quantize_text_decoder(
    execu_models: dict,
    full_model,
    device: torch.device,
    dtype,
    prefill_len: int,
    precision_name: str,
):
    """prepare_pt2e the text decoder and install it on full_model."""
    example_inputs, dynamic_example_inputs = full_model._prepare_text_input(
        device, dtype, num_tokens=prefill_len
    )
    ep = _export_text_decoder_graph(
        execu_models["text_decoder"], example_inputs, dynamic_example_inputs
    )
    quantizer = _make_enn_quantizer(precision_name)
    quantizer.add_custom_quant_annotations((custom_annotate_gemma4_text_decoder,))
    q_model = prepare_pt2e(ep.module(), quantizer)
    q_model.to(device)
    full_model.model.text_decoder = q_model
    return q_model


def _quantize_encoder(
    encoder,
    example_inputs: tuple,
    dynamic_shapes: dict,
    precision_name: str,
    device: torch.device,
    act_observer_cls=None,
):
    """Export an encoder with dynamic shapes and return its pt2e-prepared copy."""
    with torch.inference_mode():
        ep = torch.export.export(encoder, example_inputs, dynamic_shapes=dynamic_shapes)
    quantizer = _make_enn_quantizer(precision_name, act_observer_cls)
    q_model = prepare_pt2e(ep.module(), quantizer)
    q_model.to(device)
    return q_model


def _prepare_quantized_submodels(
    execu_models: dict,
    full_model,
    device: torch.device,
    dtype,
    prefill_len: int,
    max_patches: int,
    text_quantize: str | None,
    vision_quantize: str | None,
    audio_quantize: str | None,
) -> dict:
    """prepare_pt2e each requested sub-model and install it on full_model.

    Returns a dict mapping sub-model name to its prepared module; the modules
    are also installed on full_model so calibration runs through them.
    """
    submodels = {}
    if text_quantize:
        submodels["text_decoder"] = _quantize_text_decoder(
            execu_models, full_model, device, dtype, prefill_len, text_quantize
        )
    if vision_quantize:
        (example_pixels, example_positions), dynamic_shapes = (
            full_model._prepare_vision_input(device, dtype, max_patches)
        )
        submodels["vision_encoder"] = _quantize_encoder(
            execu_models["vision_encoder"],
            (example_pixels, example_positions),
            dynamic_shapes,
            vision_quantize,
            device,
        )
        full_model.model.vision_encoder = submodels["vision_encoder"]
    if audio_quantize:
        (example_features, example_mask), dynamic_shapes = (
            full_model._prepare_audio_input(device, dtype)
        )
        # A16W8 activations are int16 per-tensor affine, so the zero point can
        # exceed the fp16-exact integer range (|zp| > 2048); snap it to the
        # fp16 grid for backends that store zero points in fp16.
        submodels["audio_encoder"] = _quantize_encoder(
            execu_models["audio_encoder"],
            (example_features, example_mask),
            dynamic_shapes,
            audio_quantize,
            device,
            act_observer_cls=MinMaxObserverWithFP16ZeroPoint,
        )
        full_model.model.audio_encoder = submodels["audio_encoder"]
    return submodels


def _calibrate_quantized_submodels(
    full_model,
    processor,
    device: torch.device,
    dtype,
    max_seq_len: int,
    text_quantize: str | None,
    vision_quantize: str | None,
    audio_quantize: str | None,
) -> None:
    """Run one sample prompt per quantized modality through full_model so the
    activation observers collect statistics before convert_pt2e."""
    calibration_prompts = (
        (text_quantize, get_text_input),
        (vision_quantize, get_image_input),
        (audio_quantize, get_audio_input),
    )
    for enabled, get_messages in calibration_prompts:
        if not enabled:
            continue
        messages = get_messages()
        inputs, input_len = get_tokenize_input(messages, processor, device, dtype)
        with torch.inference_mode():
            full_model.generate(**inputs, max_new_tokens=max_seq_len - input_len)


def _quantize_and_convert_enn_models(
    execu_models: dict,
    full_model,
    processor,
    device: torch.device,
    prefill_len: int,
    max_patches: int,
    max_seq_len: int,
    text_quantize: str | None,
    vision_quantize: str | None,
    audio_quantize: str | None,
    evaluate: bool = False,
):
    """Quantize and convert the text/vision/audio ENN sub-models via pt2e.

    text_quantize / vision_quantize / audio_quantize are Precision enum
    member names (e.g. "A16W4", "A16W8"); None disables quantization for that
    sub-model. The sub-models are quantized via EnnQuantizer + prepare_pt2e
    (applied in place on full_model), calibrated with sample prompts,
    converted with convert_pt2e into execu_models, and moved to CPU when
    device != CPU.
    When evaluate is True, sample prompts are run through the host model after
    conversion.

    Returns the updated execu_models carrying the converted quantized
    sub-models. execu_models["full_model"] is left unmodified.
    """
    dtype = next(full_model.parameters()).dtype

    submodels = _prepare_quantized_submodels(
        execu_models,
        full_model,
        device,
        dtype,
        prefill_len,
        max_patches,
        text_quantize,
        vision_quantize,
        audio_quantize,
    )
    _calibrate_quantized_submodels(
        full_model,
        processor,
        device,
        dtype,
        max_seq_len,
        text_quantize,
        vision_quantize,
        audio_quantize,
    )

    device_cpu = torch.device(type="cpu")
    if device != device_cpu:
        full_model = full_model.to(device_cpu)
        full_model.target_device = device_cpu
        for name, module in submodels.items():
            submodels[name] = removing_gpu_node_in_graph(module)

    free_memory()

    if text_quantize:
        execu_models["text_decoder"] = convert_pt2e(submodels["text_decoder"])
        full_model.model.text_decoder = execu_models["text_decoder"]
    if vision_quantize:
        execu_models["vision_encoder"] = convert_pt2e(submodels["vision_encoder"])
        full_model.model.vision_encoder = execu_models["vision_encoder"]
    if audio_quantize:
        execu_models["audio_encoder"] = convert_pt2e(submodels["audio_encoder"])
        full_model.model.audio_encoder = execu_models["audio_encoder"]

    if evaluate:
        prompt_list = (
            [get_audio_input()] + get_image_input_list() + get_text_input_list()
        )
        for idx, messages in enumerate(prompt_list):
            ifm, ifm_len = get_tokenize_input(messages, processor, device_cpu, dtype)
            with torch.inference_mode():
                outputs = full_model.generate(
                    **ifm, max_new_tokens=max_seq_len - ifm_len
                )
            response = processor.decode(outputs[0], skip_special_tokens=False)
            logger.info(f"output{idx}: {response}")
    return execu_models


def _prune_quantize_flags(
    text_quantize: str | None,
    vision_quantize: str | None,
    audio_quantize: str | None,
    include_text: bool,
    include_vision: bool,
    include_audio: bool,
    text_prefill_quant_pt2: str | None,
    text_decode_quant_pt2: str | None,
    vision_quant_pt2: str | None,
    audio_quant_pt2: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not include_text or (
        text_prefill_quant_pt2 is not None and text_decode_quant_pt2 is not None
    ):
        text_quantize = None
    if not include_vision or vision_quant_pt2 is not None:
        vision_quantize = None
    if not include_audio or audio_quant_pt2 is not None:
        audio_quantize = None
    return text_quantize, vision_quantize, audio_quantize


def _get_device(device: str | None, enable_quantize: bool) -> torch.device:
    """GPU is only used for quantization calibration; export always runs on CPU."""
    device_str = str(device) if device is not None else "cpu"
    if device is None or not enable_quantize:
        if device_str != "cpu":
            logger.warning(
                f"Ignoring --device {device_str}: quantization is disabled, "
                "exporting on CPU (GPU is only used for quantization calibration)."
            )
        return torch.device("cpu")
    return torch.device(device_str)


def _quant_status(quantize: str | None, quant_pt2: str | None) -> str:
    """Describe how a component is quantized, for the quantization log line."""
    return quantize or ("pt2" if quant_pt2 else "off")


def _save_quantized_pt2(save_pt2_dir: str | None, name: str, program) -> None:
    """Save a quantized component as <name>.pt2 for later reuse."""
    if save_pt2_dir is None:
        return
    os.makedirs(save_pt2_dir, exist_ok=True)
    torch.export.save(program, os.path.join(save_pt2_dir, f"{name}.pt2"))
    logger.info(f"Saved quantized {name} to {save_pt2_dir}")


def _export_audio_component(
    programs: dict,
    checkpoint_path: str,
    execu_models: dict,
    audio_quant_pt2: str | None,
    save_pt2_dir: str | None,
) -> None:
    """Load or export the audio encoder program, saving it when requested."""
    if audio_quant_pt2 is not None:
        programs["audio_encoder"] = torch.export.load(audio_quant_pt2)
        return
    programs["audio_encoder"] = _export_audio_encoder(
        checkpoint_path, execu_models["audio_encoder"]
    )
    _save_quantized_pt2(save_pt2_dir, "audio_encoder", programs["audio_encoder"])


def _export_vision_component(
    programs: dict,
    execu_models: dict,
    max_patches: int,
    artifact_dir: str | None,
    vision_quant_pt2: str | None,
    save_pt2_dir: str | None,
) -> None:
    """Load or export the vision encoder program, saving it when requested."""
    if vision_quant_pt2 is not None:
        programs["vision_encoder"] = torch.export.load(vision_quant_pt2)
        return
    programs["vision_encoder"] = _export_vision_encoder(
        execu_models["vision_encoder"],
        execu_models["full_model"],
        max_patches=max_patches,
        artifact_dir=artifact_dir,
    )
    _save_quantized_pt2(save_pt2_dir, "vision_encoder", programs["vision_encoder"])


def _export_text_component(
    programs: dict,
    execu_models: dict,
    prefill_len: int,
    artifact_dir: str | None,
    text_prefill_quant_pt2: str | None,
    text_decode_quant_pt2: str | None,
    save_pt2_dir: str | None,
) -> None:
    """Load or export the text decoder programs, saving them when requested."""
    if text_prefill_quant_pt2 is not None:
        programs.update(
            {
                PREFILL_FORWARD: torch.export.load(text_prefill_quant_pt2),
                DECODE_FORWARD: torch.export.load(text_decode_quant_pt2),
            }
        )
        return
    text_decoder_programs = _export_text_decoder(
        execu_models["text_decoder"],
        execu_models["full_model"],
        prefill_len=prefill_len,
        artifact_dir=artifact_dir,
    )
    for graph_name in GRAPH_NAMES:
        _save_quantized_pt2(save_pt2_dir, graph_name, text_decoder_programs[graph_name])
    programs.update(text_decoder_programs)


def _export_components(
    checkpoint_path: str,
    text_quantize: str | None,
    vision_quantize: str | None,
    audio_quantize: str | None,
    max_seq_len: int,
    max_patches: int,
    include_audio: bool,
    include_speech_transform: bool,
    include_vision: bool,
    include_text: bool,
    prefill_len: int = 128,
    device: str = "cpu",
    evaluate: bool = False,
    vision_quant_pt2: str | None = None,
    audio_quant_pt2: str | None = None,
    text_prefill_quant_pt2: str | None = None,
    text_decode_quant_pt2: str | None = None,
    artifact_dir: str | None = None,
    variant: str = "e2b",
    save_pt2_dir: str | None = None,
) -> tuple[dict, dict]:
    """Export each requested component to an ExportedProgram."""
    components = [
        name
        for name, included in (
            ("speech_transform", include_speech_transform),
            ("audio_encoder", include_audio),
            ("vision_encoder", include_vision),
            ("text_decoder", include_text),
        )
        if included
    ]
    num_components = len(components)
    logger.info(f"Exporting {num_components} methods: {', '.join(components)}")

    programs = {}
    position_embeddings = None
    step = 0

    # Quantization is skipped when a component is excluded or when its
    # pre-quantized .pt2 inputs are provided instead.
    text_quantize, vision_quantize, audio_quantize = _prune_quantize_flags(
        text_quantize,
        vision_quantize,
        audio_quantize,
        include_text,
        include_vision,
        include_audio,
        text_prefill_quant_pt2,
        text_decode_quant_pt2,
        vision_quant_pt2,
        audio_quant_pt2,
    )

    logger.info(
        f"Quantization: text={_quant_status(text_quantize, text_prefill_quant_pt2)}, "
        f"vision={_quant_status(vision_quantize, vision_quant_pt2)}, "
        f"audio={_quant_status(audio_quantize, audio_quant_pt2)}"
    )

    enable_quantize = any([text_quantize, vision_quantize, audio_quantize])
    device = _get_device(device, enable_quantize)

    # Load all sub-models in one shot via Gemma4ForConditionalGeneration_ENN.
    execu_models, processor = _load_execu_models(
        checkpoint_path,
        device=device,
        max_patches=max_patches,
        max_seq_len=max_seq_len,
        variant=variant,
    )
    metadata = execu_models["full_model"].get_metadata(prefill_len)

    if enable_quantize:
        execu_models = _quantize_and_convert_enn_models(
            execu_models=execu_models,
            full_model=execu_models["full_model"],
            processor=processor,
            device=device,
            prefill_len=prefill_len,
            max_patches=max_patches,
            max_seq_len=max_seq_len,
            text_quantize=text_quantize,
            vision_quantize=vision_quantize,
            audio_quantize=audio_quantize,
            evaluate=evaluate,
        )

    if include_speech_transform:
        step += 1
        logger.info(f"[{step}/{num_components}] Exporting speech transform...")
        programs["speech_transform"] = _export_speech_transform(checkpoint_path)

    if include_audio:
        step += 1
        logger.info(f"[{step}/{num_components}] Exporting audio encoder...")
        _export_audio_component(
            programs, checkpoint_path, execu_models, audio_quant_pt2, save_pt2_dir
        )

    if include_vision:
        step += 1
        logger.info(f"[{step}/{num_components}] Exporting vision encoder...")
        _export_vision_component(
            programs,
            execu_models,
            max_patches,
            artifact_dir,
            vision_quant_pt2,
            save_pt2_dir,
        )

    if include_text:
        step += 1
        logger.info(
            f"[{step}/{num_components}] Exporting text decoder "
            f"(graphs: {', '.join(GRAPH_NAMES)})..."
        )
        _export_text_component(
            programs,
            execu_models,
            prefill_len,
            artifact_dir,
            text_prefill_quant_pt2,
            text_decode_quant_pt2,
            save_pt2_dir,
        )

        # Return the full-length RoPE tables while the full model is alive;
        # it is freed below. The runtime slices per-step cos/sin rows from
        # them to feed the graphs.
        position_embeddings = execu_models["full_model"].get_position_embeddings()

    del execu_models["full_model"]
    free_memory()

    return programs, metadata, position_embeddings


def _build_enn_partitioners(
    include_audio: bool,
    include_speech_transform: bool,
    include_vision: bool,
    include_text: bool,
    chipset: str = "E9965",
) -> dict:
    """Build per-method ENN partitioner lists."""
    from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
    from executorch.backends.samsung.serialization.compile_options import (
        gen_samsung_backend_compile_spec,
        gen_samsung_backend_compile_weight_spec,
        PerformanceMode,
        WeightSharingFlag,
    )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    compile_specs = [
        gen_samsung_backend_compile_spec(
            chipset,
            PerformanceMode.HIGH_PERFORMANCE,
        )
    ]

    enn_partitioner = EnnPartitioner(compile_specs)

    def _text_decoder_for(graph_name: str) -> list:
        weight_sharing_flag = {
            PREFILL_FORWARD: WeightSharingFlag.WEIGHT_SHARING_GEN,
            DECODE_FORWARD: WeightSharingFlag.WEIGHT_SHARING_USE,
        }[graph_name]
        specs = [
            gen_samsung_backend_compile_spec(
                chipset,
                PerformanceMode.HIGH_PERFORMANCE,
                weight_sharing_flag,
            ),
            gen_samsung_backend_compile_weight_spec(),
        ]
        return [EnnPartitioner(specs)]

    partitioners = {}
    if include_speech_transform:
        partitioners["speech_transform"] = [XnnpackPartitioner()]
    if include_audio:
        partitioners["audio_encoder"] = [enn_partitioner]
    if include_vision:
        partitioners["vision_encoder"] = [enn_partitioner]
    if include_text:
        for graph_name in GRAPH_NAMES:
            partitioners[graph_name] = _text_decoder_for(graph_name)
    return partitioners


def _build_transform_passes(
    include_audio: bool,
    include_speech_transform: bool,
    include_vision: bool,
    include_text: bool,
) -> dict:
    """Build per-method transform passes (text decoder gets bitwise lowering)."""
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.pass_base import ExportPass

    class _ReplaceBitwiseScalarPass(ExportPass):
        def __init__(self, ops_map):
            self.ops_map = ops_map
            super().__init__()

        def call_operator(self, op, args, kwargs, meta):
            if op not in self.ops_map:
                return super().call_operator(op, args, kwargs, meta)
            full_op = exir_ops.edge.aten.full.default
            scalar_as_tensor = super().call_operator(
                full_op,
                ((1,), float(args[1])),
                {
                    "dtype": args[0].to_tensor().dtype,
                    "device": args[0].to_tensor().device,
                },
                meta,
            )
            return super().call_operator(
                self.ops_map[op], (args[0], scalar_as_tensor, *args[2:]), kwargs, meta
            )

    bitwise_ops = {
        exir_ops.edge.aten.__rshift__.Scalar: exir_ops.edge.aten.bitwise_right_shift.Tensor,
        exir_ops.edge.aten.__lshift__.Scalar: exir_ops.edge.aten.bitwise_left_shift.Tensor,
        exir_ops.edge.aten.__and__.Scalar: exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.__or__.Scalar: exir_ops.edge.aten.bitwise_or.Tensor,
        exir_ops.edge.aten.__xor__.Scalar: exir_ops.edge.aten.bitwise_xor.Tensor,
    }

    text_decoder_passes = [_ReplaceBitwiseScalarPass(bitwise_ops)]

    transform_passes = {}
    if include_speech_transform:
        transform_passes["speech_transform"] = []
    if include_audio:
        transform_passes["audio_encoder"] = []
    if include_vision:
        transform_passes["vision_encoder"] = []
    if include_text:
        for graph_name in GRAPH_NAMES:
            transform_passes[graph_name] = text_decoder_passes
    return transform_passes


def export_single_pte(
    checkpoint_path: str,
    output_path: str,
    text_quantize: str | None = None,
    vision_quantize: str | None = None,
    audio_quantize: str | None = None,
    max_seq_len: int = 4096,
    max_patches: int = 1260,
    variant: str = "e2b",
    include_audio: bool = True,
    include_speech_transform: bool = True,
    include_vision: bool = True,
    include_text: bool = True,
    prefill_len: int = 128,
    device: str = "cpu",
    evaluate: bool = False,
    vision_quant_pt2: str | None = None,
    audio_quant_pt2: str | None = None,
    text_prefill_quant_pt2: str | None = None,
    text_decode_quant_pt2: str | None = None,
    chipset: str = "E9965",
    dump: bool = False,
    save_pt2_dir: str | None = None,
) -> Path:
    """Export components into a single PTE.
    By default includes all 5 methods: speech_transform, audio_encoder,
    vision_encoder, text_decoder_prefill_forward, text_decoder_decode_forward.
    Use include_audio=False, include_speech_transform=False,
    include_vision=False, or include_text=False to exclude
    components.
    """
    from executorch.backends.samsung._passes import FuseEmbeddingDequantPass
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.sym_shape_eval_pass import (
        ConstraintBasedSymShapeEvalPass,
    )
    from executorch.extension.export_util.utils import save_pte_program

    seed_everything()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    programs, metadata, position_embeddings = _export_components(
        checkpoint_path=checkpoint_path,
        text_quantize=text_quantize,
        vision_quantize=vision_quantize,
        audio_quantize=audio_quantize,
        max_seq_len=max_seq_len,
        max_patches=max_patches,
        variant=variant,
        include_audio=include_audio,
        include_speech_transform=include_speech_transform,
        include_vision=include_vision,
        include_text=include_text,
        prefill_len=prefill_len,
        device=device,
        evaluate=evaluate,
        vision_quant_pt2=vision_quant_pt2,
        audio_quant_pt2=audio_quant_pt2,
        text_prefill_quant_pt2=text_prefill_quant_pt2,
        text_decode_quant_pt2=text_decode_quant_pt2,
        artifact_dir=(str(output_path.parent) if dump else None),
        save_pt2_dir=save_pt2_dir,
    )

    logger.info("Combining into single PTE...")
    for name in programs:
        programs[name] = programs[name].run_decompositions({})
        if name in GRAPH_NAMES:
            programs[name] = FuseEmbeddingDequantPass()(programs[name])
            logger.info(f"Fused embedding dequant (int4) for {name}")

    partitioners = _build_enn_partitioners(
        include_audio, include_speech_transform, include_vision, include_text, chipset
    )
    transform_passes = _build_transform_passes(
        include_audio,
        include_speech_transform,
        include_vision,
        include_text,
    )

    from executorch.backends.samsung.utils.export_utils import get_edge_compile_config

    if include_audio and "audio_encoder" in programs:
        from executorch.backends.samsung._passes.decompose_unfold import DecomposeUnfold

        DecomposeUnfold().call(programs["audio_encoder"].graph_module)

    edge_manager = to_edge_transform_and_lower(
        programs,
        compile_config=get_edge_compile_config(),
        partitioner=partitioners,
        constant_methods=metadata,
        transform_passes=transform_passes,
    )

    et_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    output_path = Path(save_pte_program(et_program, str(output_path)))

    if position_embeddings is not None:
        save_position_embeddings(
            position_embeddings, str(output_path.parent / "pos_embedding.bin")
        )

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Single PTE exported: {output_path} ({size_mb:.1f} MB)")
    ptd_files = list(output_path.parent.glob("*.ptd"))
    if ptd_files:
        ptd_mb = sum(f.stat().st_size for f in ptd_files) / 1024 / 1024
        logger.info(f"Tensor data ({len(ptd_files)} .ptd files): {ptd_mb:.1f} MB")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Gemma 4 as a single PTE")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Local Hugging Face checkpoint directory. If omitted, --hf_model "
        "is downloaded from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default=GEMMA4_E2B_HF_MODEL,
        help="Hugging Face Hub model to download when --checkpoint_path is not "
        "given. Gemma models are gated: accept the license on the model page "
        "and log in with `hf auth login` first.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch, tag or commit of --hf_model to download.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="e2b",
        choices=["e2b"],
        help="Only e2b is currently supported",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tmp/gemma4.pte",
        help="Output path for single .pte file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length for text decoder KV cache",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        help="Text decoder quantization precision. If not set, the text "
        "decoder will not be quantized.",
        choices=[None, "A16W4"],
        type=str,
    )
    parser.add_argument(
        "--vision_quantize",
        default=None,
        help="Vision encoder quantization precision. If not set, the vision "
        "encoder will not be quantized.",
        choices=[None, "A16W8"],
        type=str,
    )
    parser.add_argument(
        "--audio_quantize",
        type=str,
        default=None,
        choices=[None, "A16W8"],
        help="Audio encoder quantization precision. If not set, the audio "
        "encoder will not be quantized.",
    )
    parser.add_argument(
        "--max_patches",
        type=int,
        default=1260,
        help="Max patches for vision encoder (default: 1260 = 140 soft tokens)",
    )
    parser.add_argument(
        "--prefill_len",
        type=int,
        default=128,
        help="Number of tokens processed per step during the text-decoder "
        "prefill graph (default: 128). Decode graph always uses 1.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        default=False,
        help="Exclude audio_encoder method to reduce PTE size.",
    )
    parser.add_argument(
        "--no-speech-transform",
        action="store_true",
        default=False,
        help="Exclude speech_transform method to reduce PTE size.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        default=False,
        help="Exclude vision_encoder method to reduce PTE size.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        default=False,
        help="Exclude text_decoder methods to reduce PTE size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="The device of model. Only used for quantization calibration; "
        "falls back to CPU when quantization is disabled.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate quantized model in samples.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        default=False,
        help="Dump float in/out tensors of vision encoder and text decoder "
        "next to the output .pte for offline comparison.",
    )
    parser.add_argument(
        "--vision_quant_pt2",
        type=str,
        default=None,
        help="Path to pre-quantized vision encoder .pt2 file. If provided, skips vision export and quantization.",
    )
    parser.add_argument(
        "--audio_quant_pt2",
        type=str,
        default=None,
        help="Path to pre-quantized audio encoder .pt2 file. If provided, skips audio export and quantization.",
    )
    parser.add_argument(
        "--text_prefill_quant_pt2",
        type=str,
        default=None,
        help="Path to pre-quantized text decoder prefill .pt2 file. If provided, skips prefill export and quantization.",
    )
    parser.add_argument(
        "--text_decode_quant_pt2",
        type=str,
        default=None,
        help="Path to pre-quantized text decoder decode .pt2 file. If provided, skips decode export and quantization.",
    )
    parser.add_argument(
        "--save_pt2_dir",
        type=str,
        default=None,
        help="Directory to save exported components as .pt2 files "
        "(audio_encoder.pt2, vision_encoder.pt2, "
        "text_decoder_prefill_forward.pt2, text_decoder_decode_forward.pt2) "
        "for later reuse.",
    )
    parser.add_argument(
        "-c",
        "--chipset",
        default="E9965",
        help="Samsung chipset, i.e. E9955, E9965, etc (only E9965 is currently supported)",
        type=str,
        choices=["E9965"],
    )
    args = parser.parse_args()

    if (args.text_prefill_quant_pt2 is None) != (args.text_decode_quant_pt2 is None):
        raise ValueError(
            "--text_prefill_quant_pt2 and --text_decode_quant_pt2 must be both "
            "provided or both omitted"
        )

    export_single_pte(
        checkpoint_path=args.checkpoint_path,
        checkpoint_path=resolve_checkpoint(
            args.checkpoint_path, args.hf_model, args.revision
        ),
        output_path=args.output_path,
        text_quantize=args.quantize,
        vision_quantize=args.vision_quantize,
        audio_quantize=args.audio_quantize,
        max_seq_len=args.max_seq_len,
        max_patches=args.max_patches,
        prefill_len=args.prefill_len,
        variant=args.variant,
        include_audio=not args.no_audio,
        include_speech_transform=not args.no_speech_transform,
        include_vision=not args.no_vision,
        include_text=not args.no_text,
        device=args.device,
        evaluate=args.evaluate,
        vision_quant_pt2=args.vision_quant_pt2,
        audio_quant_pt2=args.audio_quant_pt2,
        text_prefill_quant_pt2=args.text_prefill_quant_pt2,
        text_decode_quant_pt2=args.text_decode_quant_pt2,
        chipset=args.chipset,
        dump=args.dump,
        save_pt2_dir=args.save_pt2_dir,
    )


if __name__ == "__main__":
    main()
