# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.export._trace
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.nn.attention import SDPBackend
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForDepthEstimation,
    AutoModelForMaskedLM,
    AutoModelForSemanticSegmentation,
    AutoTokenizer,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.executorch import convert_and_export_with_cache
from transformers.modeling_utils import PreTrainedModel

from .task_registry import register_task, task_registry


@register_task("causal_lm")
def export_causal_lm(args):
    device = "cpu"
    dtype = args.dtype
    batch_size = 1
    max_length = 123
    cache_implementation = "static"
    attn_implementation = "sdpa"

    # Load and configure a HF model
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_repo,
        attn_implementation=attn_implementation,
        device_map=device,
        torch_dtype=dtype,
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_length,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_length,
            },
        ),
    )
    print(f"{model.config}")
    print(f"{model.generation_config}")

    input_ids = torch.tensor([[1]], dtype=torch.long)
    cache_position = torch.tensor([0], dtype=torch.long)

    def _get_constant_methods(model: PreTrainedModel):
        metadata = {
            "get_dtype": 5 if model.config.torch_dtype == torch.float16 else 6,
            "get_bos_id": model.config.bos_token_id,
            "get_eos_id": model.config.eos_token_id,
            "get_head_dim": model.config.hidden_size / model.config.num_attention_heads,
            "get_max_batch_size": model.generation_config.cache_config.batch_size,
            "get_max_seq_len": model.generation_config.cache_config.max_cache_len,
            "get_n_kv_heads": model.config.num_key_value_heads,
            "get_n_layers": model.config.num_hidden_layers,
            "get_vocab_size": model.config.vocab_size,
            "use_kv_cache": model.generation_config.use_cache,
        }
        return {k: v for k, v in metadata.items() if v is not None}

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():

        exported_prog = convert_and_export_with_cache(model, input_ids, cache_position)
        prog = (
            to_edge(
                exported_prog,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
                constant_methods=_get_constant_methods(model),
            )
            .to_backend(XnnpackPartitioner())
            .to_executorch(ExecutorchBackendConfig(extract_delegate_segments=True))
        )

    return model, prog


@register_task("masked_lm")
def export_masked_lm(args):
    device = "cpu"
    max_length = 64
    attn_implementation = "sdpa"

    config = AutoConfig.from_pretrained(args.hf_model_repo)
    kwargs = {}
    if hasattr(config, "use_cache"):
        kwargs["use_cache"] = True

    print(f"DEBUG: attn_implementation: {attn_implementation}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_repo)
    mask_token = tokenizer.mask_token
    print(f"Mask token: {mask_token}")
    inputs = tokenizer(
        f"The goal of life is {mask_token}.",
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        args.hf_model_repo,
        device_map=device,
        attn_implementation=attn_implementation,
        **kwargs,
    )
    print(f"{model.config}")
    print(f"{model.generation_config}")

    # pre-autograd export. eventually this will become torch.export
    exported_program = torch.export.export_for_training(
        model,
        args=(inputs["input_ids"],),
        kwargs={"attention_mask": inputs["attention_mask"]},
        strict=True,
    )

    return model, to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _skip_dim_order=True,
        ),
    ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))


@register_task("semantic_segmentation")
def export_semantic_segmentation(args):
    import requests
    from PIL import Image

    device = "cpu"
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.hf_model_repo,
        device_map=device,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.hf_model_repo,
        device_map=device,
    )
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors="pt")

    exported_program = torch.export.export_for_training(
        model,
        args=(inputs["pixel_values"],),
        strict=True,
    )

    return model, to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _skip_dim_order=True,
        ),
    ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))


@register_task("depth_estimation")
def export_depth_estimation(args):
    import requests
    from PIL import Image

    device = "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(
        args.hf_model_repo,
        device_map=device,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.hf_model_repo,
        device_map=device,
    )
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors="pt")

    exported_program = torch.export.export_for_training(
        model,
        args=(inputs["pixel_values"],),
        strict=True,
    )

    return model, to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _skip_dim_order=True,
        ),
    ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hfm",
        "--hf_model_repo",
        required=True,
        default=None,
        help="a valid huggingface model repo name",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="specify the dtype for loading the model",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        required=False,
        default=None,
        help="output name of the exported model",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=list(task_registry.keys()),
        default="causal_lm",
        help=f"type of task of the model to load from huggingface. supported tasks: {task_registry.keys()}",
    )

    args = parser.parse_args()
    try:
        model, prog = task_registry[args.task](args)
    except AttributeError:
        raise ValueError(f"Unsupported task type {args.task}")

    out_name = args.output_name if args.output_name else model.config.model_type
    filename = os.path.join("./", f"{out_name}.pte")
    with open(filename, "wb") as f:
        prog.write_to_file(f)
        print(f"Saved exported program to {filename}")


if __name__ == "__main__":
    main()
