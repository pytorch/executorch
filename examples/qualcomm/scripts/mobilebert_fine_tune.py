# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from multiprocessing.connection import Client

import numpy as np

import torch
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
    SimpleADB,
)
from transformers import BertTokenizer, MobileBertForSequenceClassification


def evaluate(model, data_val):
    predictions, true_vals = [], []
    for data in data_val:
        inputs = {
            "input_ids": data[0],
            "attention_mask": data[1],
            "labels": data[2],
        }
        logits = model(**inputs)[1].detach().numpy()
        label_ids = inputs["labels"].numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(true_vals, axis=0),
    )


def accuracy_per_class(preds, goldens, labels):
    labels_inverse = {v: k for k, v in labels.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    goldens_flat = goldens.flatten()

    result = {}
    for golden in np.unique(goldens_flat):
        pred = preds_flat[goldens_flat == golden]
        true = goldens_flat[goldens_flat == golden]
        result.update({labels_inverse[golden]: [len(pred[pred == golden]), len(true)]})

    return result


def get_dataset(data_val):
    # prepare input data
    inputs, input_list = [], ""
    for index, data in enumerate(data_val):
        inputs.append(tuple(data[:2]))
        input_text = " ".join(
            [f"input_{index}_{i}.raw" for i in range(len(inputs[-1]))]
        )
        input_list += f"{input_text}\n"

    return inputs, input_list


def get_fine_tuned_mobilebert(artifacts_dir, pretrained_weight):
    from io import BytesIO

    import pandas as pd
    import requests
    from sklearn.model_selection import train_test_split
    from torch.utils.data import (
        DataLoader,
        RandomSampler,
        SequentialSampler,
        TensorDataset,
    )
    from tqdm import tqdm
    from transformers import get_linear_schedule_with_warmup

    # grab dataset
    url = (
        "https://raw.githubusercontent.com/susanli2016/NLP-with-Python"
        "/master/data/title_conference.csv"
    )
    content = requests.get(url, allow_redirects=True).content
    data = pd.read_csv(BytesIO(content))

    # get training / validation data
    labels = {key: index for index, key in enumerate(data.Conference.unique())}
    data["label"] = data.Conference.replace(labels)

    train, val, _, _ = train_test_split(
        data.index.values,
        data.label.values,
        test_size=0.15,
        random_state=42,
        stratify=data.label.values,
    )

    data["data_type"] = ["not_set"] * data.shape[0]
    data.loc[train, "data_type"] = "train"
    data.loc[val, "data_type"] = "val"
    data.groupby(["Conference", "label", "data_type"]).count()

    # get pre-trained mobilebert
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
    )
    model = MobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        num_labels=len(labels),
        return_dict=False,
    )

    # tokenize dataset
    encoded_data_train = tokenizer.batch_encode_plus(
        data[data.data_type == "train"].Title.values,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded_data_val = tokenizer.batch_encode_plus(
        data[data.data_type == "val"].Title.values,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids_train = encoded_data_train["input_ids"]
    attention_masks_train = encoded_data_train["attention_mask"]
    labels_train = torch.tensor(data[data.data_type == "train"].label.values)

    input_ids_val = encoded_data_val["input_ids"]
    attention_masks_val = encoded_data_val["attention_mask"]
    labels_val = torch.tensor(data[data.data_type == "val"].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    batch_size, epochs = 3, 5
    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size,
    )
    dataloader_val = DataLoader(
        dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=batch_size,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs
    )

    # start training
    if not pretrained_weight:
        for epoch in range(1, epochs + 1):
            loss_train_total = 0
            print(f"epoch {epoch}")

            for batch in tqdm(dataloader_train):
                model.zero_grad()
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                }
                loss = model(**inputs)[0]
                loss_train_total += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            torch.save(
                model.state_dict(),
                f"{artifacts_dir}/finetuned_mobilebert_epoch_{epoch}.model",
            )

    model.load_state_dict(
        torch.load(
            (
                f"{artifacts_dir}/finetuned_mobilebert_epoch_{epochs}.model"
                if pretrained_weight is None
                else pretrained_weight
            ),
            map_location=torch.device("cpu"),
        ),
    )

    return model.eval(), dataloader_val, batch_size, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./mobilebert_fine_tune",
        default="./mobilebert_fine_tune",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--build_folder",
        help="path to cmake binary directory for android, e.g., /path/to/build_android",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="serial number for android device communicated via ADB.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-H",
        "--host",
        help="hostname where android device is connected.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="SoC model of current device. e.g. 'SM8550' for Snapdragon 8 Gen 2.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--pretrained_weight",
        help="Location of pretrained weight",
        default="",
        type=str,
    )
    parser.add_argument(
        "--ip",
        help="IPC address for delivering execution result",
        default="",
        type=str,
    )
    parser.add_argument(
        "--port",
        help="IPC port for delivering execution result",
        default=-1,
        type=int,
    )

    # QNN_SDK_ROOT might also be an argument, but it is used in various places.
    # So maybe it's fine to just use the environment.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    if "LD_LIBRARY_PATH" not in os.environ:
        print(
            "[Warning] LD_LIBRARY_PATH is not set. If errors like libQnnHtp.so "
            "not found happen, please follow setup.md to set environment."
        )
    else:
        print(f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    pte_filename = "mb_qnn"
    model, data_val, batch_size, labels = get_fine_tuned_mobilebert(
        args.artifact, args.pretrained_weight
    )
    inputs, input_list = get_dataset(data_val)

    build_executorch_binary(
        model,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        None,
        use_fp16=True,
    )

    # setup required paths accordingly
    # qnn_sdk       : QNN SDK path setup in environment variable
    # artifact_path : path where artifacts were built
    # pte_path      : path where executorch binary was stored
    # device_id     : serial number of android device
    # workspace     : folder for storing artifacts on android device
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        artifact_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
    )
    adb.push(inputs=inputs, input_list=input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    # get torch cpu result
    cpu_preds, true_vals = evaluate(model, data_val)
    cpu_result = accuracy_per_class(cpu_preds, true_vals, labels)

    # get QNN HTP result
    htp_preds = []
    for i in range(len(data_val)):
        result = np.fromfile(
            os.path.join(output_data_folder, f"output_{i}_0.raw"),
            dtype=np.float32,
        )
        htp_preds.append(result.reshape(batch_size, -1))

    htp_result = accuracy_per_class(
        np.concatenate(htp_preds, axis=0), true_vals, labels
    )

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"CPU": cpu_result, "HTP": htp_result}))
    else:
        for target in zip(["CPU", "HTP"], [cpu_result, htp_result]):
            print(f"\n[{target[0]}]")
            for k, v in target[1].items():
                print(f"{k}: {v[0]}/{v[1]}")
