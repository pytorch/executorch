# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import tqdm
from datasets import DatasetDict, load_dataset

from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoImageProcessor
from transformers.models.vit.modeling_vit import ViTForImageClassification


def make_transform(preprocessor):
    def transform(batch):
        img = [item.convert("RGB") for item in batch["image"]]
        inputs = preprocessor(img, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].unsqueeze(0),
            "labels": batch["label"],
        }

    return transform


def quantize_model(model, quantizer, calibration_data):
    example_input = calibration_data[0]["pixel_values"]

    exported_model = torch.export.export(
        model,
        (example_input,),
    ).module()

    quantize = prepare_pt2e(exported_model, quantizer)

    print("\nCalibrating the model...")
    for example in tqdm.tqdm(calibration_data):
        quantize(example["pixel_values"])

    pt2e_deit = convert_pt2e(quantize)

    return torch.export.export(
        pt2e_deit,
        (example_input,),
    )


def measure_accuracy(quantized_model, test_set):
    examples = 0
    correct = 0

    print("\nMeasuring accuracy on the test set...")
    tbar = tqdm.tqdm(test_set)
    for example in tbar:
        img = example["pixel_values"]
        output = quantized_model(img)
        output = output.logits.argmax(dim=-1).item()

        if output == example["labels"]:
            correct += 1
        examples += 1
        accuracy = correct / examples

        tbar.set_description(f"Accuracy: {accuracy:.4f}")

    print(f"Top-1 accuracy on {examples} test samples: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Export ViT model")
    argparser.add_argument(
        "--model-path",
        type=str,
        default="./deit-tiny-oxford-pet/final_model",
        required=True,
        help="Path to the fine-tuned ViT model.",
    )
    argparser.add_argument(
        "--output-path",
        type=str,
        default="./deit_quantized_exported.pte",
        help="Path to save the exported quantized model.",
    )
    argparser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=300,
        help="Number of samples to use for calibration.",
    )
    argparser.add_argument(
        "--num-test-samples",
        type=int,
        default=100,
        help="Number of samples to use for testing accuracy.",
    )
    args = argparser.parse_args()

    deit = ViTForImageClassification.from_pretrained(
        args.model_path,
        num_labels=37,
        ignore_mismatched_sizes=True,
    ).eval()
    image_preprocessor = AutoImageProcessor.from_pretrained(
        "facebook/deit-tiny-patch16-224", use_fast=True
    )

    compile_spec = EthosUCompileSpec(
        target="ethos-u85-256",
        memory_mode="Shared_Sram",
    )

    quantizer = EthosUQuantizer(compile_spec)
    operator_config = get_symmetric_quantization_config()
    quantizer.set_global(operator_config)

    ds = load_dataset("timm/oxford-iiit-pet")

    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )
    dataset = dataset.with_transform(make_transform(image_preprocessor))

    with torch.no_grad():
        quantized_deit = quantize_model(
            deit,
            quantizer,
            dataset["train"].take(args.num_calibration_samples),
        )
        measure_accuracy(
            quantized_deit.module(), dataset["test"].take(args.num_test_samples)
        )

    partition = EthosUPartitioner(compile_spec)
    edge_encoder = to_edge_transform_and_lower(
        programs=quantized_deit,
        partitioner=[partition],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )
    edge_manager = edge_encoder.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    save_pte_program(edge_manager, args.output_path)
    print(f"\nExported model saved to {args.output_path}")
