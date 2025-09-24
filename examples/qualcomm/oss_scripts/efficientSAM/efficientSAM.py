# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import os
import zipfile
from multiprocessing.connection import Client
from typing import Callable, List

import numpy as np
import torch
from executorch.backends.qualcomm._passes import ExpandBroadcastTensorShape
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.utils.constants import QCOM_PASS_ACTIVATE_KEY
from executorch.examples.qualcomm.oss_scripts.efficientSAM.source_transformation import (
    replace_maskdecoder_with_custom_op,
    replace_pos_emb_with_custom_op,
)

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    class_agnostic_mIoU,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from PIL import Image, ImageDraw
from scipy.ndimage import label
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def load_dataset(dataset_path):
    image_shape = (224, 224)
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_shape),
            transforms.ToTensor(),
        ]
    )
    imagenet_data = datasets.ImageFolder(dataset_path, transform=preprocess)

    return list(imagenet_data)


class EfficientSAMDataset(Dataset):
    def __init__(self, dataset_path, data_size=1) -> None:
        self.to_tensor = transforms.ToTensor()
        dataset = load_dataset(dataset_path)
        self.inputs = self.get_val_dataset(dataset, data_size)
        self.data_size = data_size

    def get_val_dataset(self, dataset, data_size):
        imgs, pt_prompts, pt_labels = [], [], []
        for i, data in enumerate(dataset):
            if i >= data_size:
                break
            img = data[0]
            h, w = img.shape[-2:]

            # Assuming the main object usually appears in the middle of the image, this default value is set for better demo visualization.
            # Users can modify/add the point prompt here.
            pt_prompt = torch.tensor([[w / 2, (h * 2 / 3)]], dtype=torch.float32)[
                None, ...
            ]
            # Users can increase the tensor size by adding more labels (0 for negative samples, 1 for positive samples) to label the corresponding points.
            # The default label is [[1]], indicating that the point is a positive sample.
            pt_label = torch.tensor([[1]], dtype=torch.float32)

            imgs.append(img)
            pt_prompts.append(pt_prompt)
            pt_labels.append(pt_label)

        imgs = torch.stack(imgs)
        pt_prompts = torch.stack(pt_prompts)
        pt_labels = torch.stack(pt_labels)
        inputs = (imgs, pt_prompts, pt_labels)
        return inputs

    def __getitem__(self, idx):
        return self.inputs[0][idx], self.inputs[1][idx], self.inputs[2][idx]

    def __len__(self):
        return self.data_size


def get_dataset(dataset_path, data_size=1):

    dataset = EfficientSAMDataset(dataset_path, data_size=data_size)
    dataloader = DataLoader(dataset)

    # prepare input data
    inputs = []
    for index, data in enumerate(dataloader):
        if index >= data_size:
            break
        inputs.append(tuple(data))

    return inputs


def source_transform(
    model, transforms: List[Callable[[torch.nn.Module], torch.nn.Module]]
):
    for transform in transforms:
        model = transform(model)
    return model


def get_instance(args):
    import sys

    sys.path.insert(0, args.oss_repo)
    from efficient_sam.efficient_sam import build_efficient_sam

    ckpt = args.pretrained_weight
    file_path, file_extension = os.path.splitext(ckpt)
    file_dir, filename = os.path.split(file_path)

    if file_extension == ".zip":
        with zipfile.ZipFile(ckpt, "r") as zip_ref:
            zip_ref.extractall(file_dir)
        ckpt = file_path
        filename = os.path.splitext(filename)[0]

    model_arch = filename.split("_")[-1]

    if model_arch == "vitt":
        encoder_patch_embed_dim, encoder_num_heads = (192, 3)
    elif model_arch == "vits":
        encoder_patch_embed_dim, encoder_num_heads = (384, 6)
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    model = build_efficient_sam(
        encoder_patch_embed_dim=encoder_patch_embed_dim,
        encoder_num_heads=encoder_num_heads,
        checkpoint=ckpt,
    ).eval()

    return model


def generate_mask(predicted_logits, predicted_iou):
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )

    # The masks are already sorted by their predicted IOUs.
    # We use the first mask.
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    return mask


def save_mask(mask, input, save_path):
    image, prompt, pt_label = input
    original_image_tensor = image[0]

    # Convert tensor to numpy array if necessary
    if not isinstance(original_image_tensor, np.ndarray):
        original_image_tensor = original_image_tensor.detach().numpy()

    # Transpose if the image has 3 channels
    if original_image_tensor.shape[0] == 3:
        original_image_tensor = original_image_tensor.transpose(1, 2, 0)

    original_img = Image.fromarray(
        (original_image_tensor * 255).astype(np.uint8)
    ).convert("RGBA")

    # Create an empty RGBA image for the mask
    mask_img = np.ones((mask.shape[0], mask.shape[1], 4))
    mask_img[:, :, 3] = 0

    colors = [
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.5],
        [0, 0, 1, 0.5],
        [1, 1, 0, 0.5],
        [1, 0, 1, 0.5],
        [0, 1, 1, 0.5],
    ]

    # Apply mask
    labeled_mask, num_feature = label(mask)
    for i in range(1, num_feature + 1):
        mask_img[labeled_mask == i] = colors[(i - 1) % len(colors)]

    mask_img = Image.fromarray((mask_img * 255).astype(np.uint8), "RGBA")

    # Combine original image with mask
    combined_img = Image.alpha_composite(original_img, mask_img)

    # Draw prompts point ("green" for positive samples, "red" for negative samples)
    draw = ImageDraw.Draw(combined_img)
    for pt, l in zip(prompt[0][0], pt_label[0][0]):
        color = "green" if l else "red"
        point_size = 3
        x1, y1 = max(0, int(pt[0]) - point_size), max(0, int(pt[1]) - point_size)
        x2, y2 = min(combined_img.size[0], int(pt[0]) + point_size), min(
            combined_img.size[1], int(pt[1]) + point_size
        )
        draw.ellipse((x1, y1, x2, y2), fill=color, outline=color)

    combined_img.save(save_path)


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    os.makedirs(args.artifact, exist_ok=True)

    data_size = 1
    inputs = get_dataset(args.dataset, data_size)
    assert args.pretrained_weight, "Checkpoint params can't be empty"

    # Get the EfficientSAM model.
    model = get_instance(args)
    model = source_transform(
        model,
        [
            replace_maskdecoder_with_custom_op,
            replace_pos_emb_with_custom_op,
        ],
    )

    pte_filename = "efficientSAM_qnn"

    # lower to QNN
    passes_job = get_capture_program_passes()
    passes_job[ExpandBroadcastTensorShape][QCOM_PASS_ACTIVATE_KEY] = True
    build_executorch_binary(
        model,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        passes_job=passes_job,
        shared_buffer=args.shared_buffer,
    )

    if args.compile_only:
        return

    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{pte_filename}"
    pte_path = f"{args.artifact}/{pte_filename}.pte"

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=pte_path,
        workspace=workspace,
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        for i, f in enumerate(sorted(os.listdir(output_data_folder))):
            filename = os.path.join(output_data_folder, f)
            output = np.fromfile(filename, dtype=np.float32)
            output_shape = [1, 1, 3] if i % 2 else [1, 1, 3, 224, 224]
            output = torch.from_numpy(output).reshape(output_shape)
            outputs.append(output)

    adb.pull(output_path=args.artifact, callback=post_process)

    # MIoU analysis
    miou = 0
    targets = [model(img, pt, pt_label) for img, pt, pt_label in inputs]
    for i in range(data_size):
        pred_mask = generate_mask(outputs[i * 2], outputs[i * 2 + 1])
        save_mask(pred_mask, inputs[i], f"{args.artifact}/output_{i}.png")
        target_mask = generate_mask(targets[i][0], targets[i][1])
        miou += class_agnostic_mIoU([pred_mask], [target_mask])
    miou /= data_size

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"MIoU": miou}))
    else:
        print(f"MIoU->{miou}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./EfficientSAM_qnn",
        default="./EfficientSAM_qnn",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--pretrained_weight",
        help="Path to ESAM checkpoint, such as ./efficient_sam_vitt.pt or ./efficient_sam_vits.pt.zip",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help=(
            "path to the validation folder of ImageNet dataset. "
            "e.g. --dataset imagenet-mini/val "
            "for https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)"
        ),
        type=str,
        required=True,
    )

    parser.add_argument(
        "--oss_repo",
        help="Path to clone https://github.com/yformer/EfficientSAM",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    args.validate(args)
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
