# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from multiprocessing.connection import Client

import numpy as np

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    skip_annotation,
)
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    QnnPartitioner,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir import to_edge
from transformers import BertTokenizer, MobileBertForSequenceClassification


def evaluate(model, data_val):
    predictions, true_vals = [], []
    for data in data_val:
        inputs = {
            "input_ids": data[0].to(torch.long),
            "attention_mask": data[1].to(torch.long),
            "labels": data[2].to(torch.long),
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
    # max_position_embeddings defaults to 512
    position_ids = torch.arange(512).expand((1, -1)).to(torch.int32)
    for index, data in enumerate(data_val):
        data = [d.to(torch.int32) for d in data]
        # input_ids, attention_mask, token_type_ids, position_ids
        inputs.append(
            (
                *data[:2],
                torch.zeros(data[0].size(), dtype=torch.int32),
                position_ids[:, : data[0].shape[1]],
            )
        )
        input_text = " ".join(
            [f"input_{index}_{i}.raw" for i in range(len(inputs[-1]))]
        )
        input_list += f"{input_text}\n"

    return inputs, input_list


def get_fine_tuned_mobilebert(artifacts_dir, pretrained_weight, batch_size):
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

    epochs = 5
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
            weights_only=True,
        ),
    )

    return model.eval(), dataloader_val, labels


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    batch_size, pte_filename = 1, "ptq_mb_qnn"
    model, data_val, labels = get_fine_tuned_mobilebert(
        args.artifact, args.pretrained_weight, batch_size
    )
    inputs, input_list = get_dataset(data_val)

    try:
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")
    except:
        raise AssertionError(
            f"No support for quant type {args.ptq}. Support 8a8w, 16a16w and 16a4w."
        )

    if args.use_fp16:
        quant_dtype = None
        pte_filename = "mb_qnn"
        build_executorch_binary(
            model,
            inputs[0],
            args.model,
            f"{args.artifact}/{pte_filename}",
            inputs,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            quant_dtype=quant_dtype,
            shared_buffer=args.shared_buffer,
        )
    else:

        def calibrator(gm):
            for input in inputs:
                gm(*input)

        quantizer = make_quantizer(quant_dtype=quant_dtype)
        backend_options = generate_htp_compiler_spec(quant_dtype is not None)
        partitioner = QnnPartitioner(
            generate_qnn_executorch_compiler_spec(
                soc_model=getattr(QcomChipset, args.model),
                backend_options=backend_options,
            ),
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
        )
        # skip embedding layer cause it's quantization sensitive
        graph_module, _ = skip_annotation(
            nn_module=model,
            quantizer=quantizer,
            partitioner=partitioner,
            sample_input=inputs[0],
            calibration_cb=calibrator,
            fp_node_op_set={torch.ops.aten.embedding.default},
        )
        # lower all graph again, the skipped operators will be left in CPU
        exec_prog = to_edge(
            torch.export.export(graph_module, inputs[0]),
        ).to_executorch()

        with open(f"{args.artifact}/{pte_filename}.pte", "wb") as file:
            file.write(exec_prog.buffer)

    if args.compile_only:
        sys.exit(0)

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
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


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./mobilebert_fine_tune",
        default="./mobilebert_fine_tune",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--pretrained_weight",
        help="Location of pretrained weight",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-F",
        "--use_fp16",
        help="If specified, will run in fp16 precision and discard ptq setting",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 8bits activation and 8bits weight. Support 8a8w, 16a16w and 16a4w.",
        default="8a8w",
    )

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
