# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from evaluate import load as load_metric
from transformers import AutoImageProcessor, set_seed, Trainer, TrainingArguments

from transformers.models.vit.modeling_vit import ViTForImageClassification


def make_transform(preprocessor):
    def transform(batch):
        img = [item.convert("RGB") for item in batch["image"]]
        inputs = preprocessor(img, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"],
            "labels": batch["label"],
        }

    return transform


def make_compute_metrics(accuracy_metric):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    return compute_metrics


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(42)

    argparser = argparse.ArgumentParser(
        description="Fine-tune DeIT model on Oxford-IIIT Pet dataset"
    )
    argparser.add_argument(
        "--output-dir",
        type=str,
        default="./deit-tiny-oxford-pet",
        help="Directory to save the trained model",
    )
    argparser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs"
    )
    args = argparser.parse_args()
    ds = load_dataset("timm/oxford-iiit-pet")

    # Create the mappings between labels and IDs
    labels = ds["train"].features["label"].names
    ids2label = dict(enumerate(labels))
    label2ids = {l: i for i, l in enumerate(labels)}

    deit = ViTForImageClassification.from_pretrained(
        "facebook/deit-tiny-patch16-224",
        num_labels=37,
        ignore_mismatched_sizes=True,
        id2label=ids2label,
        label2id=label2ids,
    )
    image_preprocessor = AutoImageProcessor.from_pretrained(
        "facebook/deit-tiny-patch16-224", use_fast=True
    )

    # Create a validation set by splitting the training set into two parts
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": ds["test"],
        }
    )
    dataset = dataset.with_transform(make_transform(image_preprocessor))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,
        eval_strategy="steps",
        num_train_epochs=args.num_epochs,
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        report_to="none",
    )

    accuracy_metric = load_metric("accuracy")
    trainer = Trainer(
        model=deit,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=make_compute_metrics(accuracy_metric),
    )

    print("\n Starting training DEiT Tiny on Oxford-IIIT Pet dataset...")
    trainer.train()

    print("\nEvaluating the model on the test set...")
    result = trainer.evaluate(dataset["test"])
    print(f"Test set accuracy: {result['eval_accuracy']:.4f}")

    final_model_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"\nTrained model saved to {final_model_path}")
