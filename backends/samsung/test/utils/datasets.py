# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Literal, Optional, Tuple

import torch
import torchvision.transforms.v2 as vision_transform_v2
from executorch.backends.samsung.test.utils.utils import GreedyLM
from torchsr import transforms as sr_transforms
from torchvision import transforms as vision_transforms
from torchvision.datasets import ImageFolder, VOCSegmentation


def get_quant_test_data_classify(
    data_dir: str,
    calinum=100,
    testnum=500,
    transform_compose: Optional[vision_transforms.Compose] = None,
) -> Tuple:
    """
    Generate test data for quantization model

    :param data_dir: Dir of dataset. Structure should be imagenet-like
    :param calinum: Number of calibration data. Default 100
    :param testnum: Number of test data. Default 500
    :param transform_compose: Transforms to be applied to data.

        Default:
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dim
        ]
    :type data_dir: str
    :type calinum: int
    :type testnum: int
    :type transform_compose: transforms.Compose | None
    :return: (example_input, test_data)
    """
    if not transform_compose:
        transform_compose = vision_transforms.Compose(
            [
                vision_transforms.Resize((256, 256)),
                vision_transforms.CenterCrop(224),
                vision_transforms.ToTensor(),
                vision_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                vision_transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dim
            ]
        )
    dataset = ImageFolder(root=data_dir, transform=transform_compose)
    cali_data = [(dataset[i][0],) for i in range(min(calinum, len(dataset)))]
    test_data = [dataset[i] for i in range(min(testnum, len(dataset)))]
    example_input = (dataset[0][0],)
    return example_input, cali_data, test_data


def get_quant_test_data_super_resolution(
    root_dir: str,
    dataset_name: Literal["B100", "Set5", "Set14", "Urban100"],
    calinum=100,
    testnum=500,
    transform_compose: Optional[sr_transforms.Compose] = None,
) -> Tuple:
    """
    Generate test data for quantization model

    :param root_dir: Dir of dataset. The real dataset should be in root_dir/SRBenchmarks/benchmark/
    :param dataset_name: data_set name
    :param testnum: Number of test data. Default 500
    :param transform_compose: Transforms to be applied to data.
        Default:
        transform_compose = transforms.Compose(
            [transforms.ToTensor()] # Convert Pillows Image to tensor
        )
    :type root_dir: str
    :type dataset_name: "B100"|"Set5"|"Set14"|"Urban100"
    :type calinum: int
    :type testnum: int
    :type transform_compose: transforms.Compose | None
    :return: (example_input, cali_data, test_data)
    """

    class SrResize:
        def __init__(self, expected_size: List[List[int]]):
            self.expected_size = expected_size

        def __call__(self, x):
            return (
                x[0].resize(self.expected_size[0]),
                x[1].resize(self.expected_size[1]),
            )

    class SrUnsqueeze:
        def __call__(self, x):
            return (
                x[0].unsqueeze(0),
                x[1].unsqueeze(0),
            )

    if not transform_compose:
        transform_compose = sr_transforms.Compose(
            [
                SrResize([[448, 448], [224, 224]]),
                sr_transforms.ToTensor(),  # Convert Pillows Image to tensor
                SrUnsqueeze(),
            ]
        )
    from torchsr.datasets import B100, Set14, Set5, Urban100

    dataset_cls_map = {
        "B100": B100,
        "Set5": Set5,
        "Set14": Set14,
        "Urban100": Urban100,
    }

    dataset_cls = dataset_cls_map.get(dataset_name)
    assert dataset_cls
    dataset = dataset_cls(root=root_dir, transform=transform_compose, scale=2)
    calib_data = [(dataset[i][1],) for i in range(min(calinum, len(dataset)))]
    test_data = [
        (dataset[i][1], dataset[i][0]) for i in range(min(testnum, len(dataset)))
    ]
    example_input = (dataset[0][1],)
    return example_input, calib_data, test_data


def get_quant_test_data_segmentation(
    data_dir: str,
    calinum=100,
    testnum=500,
    input_transform_compose: Optional[vision_transform_v2.Compose] = None,
    target_transform_compose: Optional[vision_transform_v2.Compose] = None,
):
    if not input_transform_compose:
        input_transform_compose = vision_transform_v2.Compose(
            [
                vision_transform_v2.Resize([224, 224]),
                vision_transform_v2.ToImage(),
                vision_transform_v2.ToDtype(torch.float32, scale=True),
                vision_transform_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                vision_transform_v2.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dim
            ]
        )
    if not target_transform_compose:
        target_transform_compose = vision_transform_v2.Compose(
            [
                vision_transform_v2.Resize([224, 224]),
                vision_transform_v2.ToImage(),
                vision_transform_v2.ToDtype(torch.long, scale=False),
                vision_transform_v2.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dim
            ]
        )
    voc_dataset = VOCSegmentation(
        data_dir,
        "2012",
        "val",
        transform=input_transform_compose,
        target_transform=target_transform_compose,
    )
    calib_data = [(voc_dataset[i][0],) for i in range(min(calinum, len(voc_dataset)))]
    test_data = [voc_dataset[i] for i in range(min(testnum, len(voc_dataset)))]
    example_input = (voc_dataset[0][0],)
    return example_input, calib_data, test_data


def _get_voice_dataset(
    data_size: int, data_dir: str, labels: List[str], fixed_token_num: int
):
    from torch.utils.data import DataLoader
    from torchaudio.datasets import LIBRISPEECH

    def collate_fun(batch, encode_fn, mode="train"):
        waves = []
        text_ids = []
        input_lengths = []
        output_lengths = []

        if mode == "train":
            shifts = torch.randn(len(batch)) > 0.0

        for i, (wave, _, text, *_) in enumerate(batch):
            if mode == "train" and shifts[i]:
                wave = wave[:, 160:]
            waves.append(wave[0])
            ids = torch.LongTensor(encode_fn(text))
            text_ids.append(ids)
            input_lengths.append(wave.size(1) // 320)
            output_lengths.append(len(ids))

        waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).unsqueeze(1)
        labels = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True)

        return waves, labels, input_lengths, output_lengths

    lm = GreedyLM(labels)

    testset_url = "test-clean"
    # testset_url = 'test-clean'
    dataset = LIBRISPEECH(data_dir, url=testset_url)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: collate_fun(x, lm.encode, "valid"),
    )
    # prepare input data
    inputs, targets = [], []
    in_lens, tar_lens = [], []

    def _loader():
        for waves, labels, inputs_len, targets_len in data_loader:
            if inputs_len[0] >= fixed_token_num:
                continue
            zero_padding = torch.zeros([1, 1, fixed_token_num * 320 - waves.shape[2]])
            waves = torch.concat((waves, zero_padding), axis=2)
            yield waves, labels, [fixed_token_num + 1], targets_len

    for i, (waves, labels, inputs_len, targets_len) in enumerate(
        _loader()
    ):  # waves, labels, input_lens, output_lens
        inputs.append(waves)
        targets.append(labels)
        in_lens.append(inputs_len)
        tar_lens.append(targets_len)
        if i >= data_size:
            break

    return inputs, targets, in_lens, tar_lens


def get_quant_test_data_voice(
    data_dir: str,
    calinum=100,
    testnum=500,
    fixed_out_token=300,
    labels=None,
):
    if labels is None:
        labels = [" ", *"abcdefghijklmnopqrstuvwxyz", "'", "*"]
    dataset = _get_voice_dataset(
        max(testnum, calinum), data_dir, labels, fixed_out_token
    )
    calib_data = [(dataset[0][i],) for i in range(min(calinum, len(dataset[0])))]
    test_data = [
        (dataset[0][i], (dataset[1][i], dataset[2][i], dataset[3][i]))
        for i in range(min(testnum, len(dataset[0])))
    ]
    example_input = (dataset[0][0],)
    return example_input, calib_data, test_data
