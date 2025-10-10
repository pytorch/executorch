# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List

import torch

from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer import Precision
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    quantize_module,
    to_edge_transform_and_lower_to_enn,
)
from executorch.examples.models.wav2letter import Wav2LetterModel
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program


class DataManager:
    class Encoder:
        def __init__(self, vocab, blank_label="*"):
            self.vocab = vocab
            self.char_to_id = {c: i for i, c in enumerate(vocab)}
            self.blank_label = blank_label

        def encode(self, text):
            return [self.char_to_id[c] for c in text.lower()]

    @classmethod
    def _get_voice_dataset(
        cls, data_size: int, data_dir: str, labels: List[str], fixed_token_num: int
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

            waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).unsqueeze(
                1
            )
            labels = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True)

            return waves, labels, input_lengths, output_lengths

        encoder = cls.Encoder(labels)

        testset_url = "test-clean"
        dataset = LIBRISPEECH(data_dir, url=testset_url)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: collate_fun(x, encoder.encode, "valid"),
        )
        # prepare input data
        inputs, targets = [], []
        in_lens, tar_lens = [], []

        def _loader():
            for waves, labels, inputs_len, targets_len in data_loader:
                if inputs_len[0] >= fixed_token_num:
                    continue
                zero_padding = torch.zeros(
                    [1, 1, fixed_token_num * 320 - waves.shape[2]]
                )
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

    @classmethod
    def get_dataset(
        cls,
        data_dir: str,
        calinum=100,
        fixed_out_token=300,
        labels=None,
    ):
        if labels is None:
            labels = [" ", *"abcdefghijklmnopqrstuvwxyz", "'", "*"]
        dataset = cls._get_voice_dataset(calinum, data_dir, labels, fixed_out_token)
        example_input = [(dataset[0][i],) for i in range(min(calinum, len(dataset[0])))]
        return example_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--chipset",
        default="E9955",
        help="Samsung chipset, i.e. E9945, E9955, etc",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help=(
            "path to the validation folder of ImageNet dataset. "
            "e.g. --dataset imagenet-mini/val "
            "for https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)"
        ),
        type=str,
    )

    parser.add_argument(
        "-p",
        "--precision",
        default=None,
        help=("Quantizaiton precision. If not set, the model will not be quantized."),
        choices=[None, "A8W8"],
        type=str,
    )

    parser.add_argument(
        "-cn",
        "--calibration_number",
        default=100,
        help=(
            "Assign the number of data you want "
            "to use for calibrating the quant params."
        ),
        type=int,
    )

    parser.add_argument(
        "--dump",
        default=False,
        const=True,
        nargs="?",
        help=("Whether to dump all outputs. If not set, we only dump pte."),
        type=bool,
    )

    parser.add_argument(
        "-w",
        "--weight",
        default=None,
        help="Absolute path of retrained w2l weight (With .pt format), the vocab size should 29",
        type=str,
    )

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. ",
        default="./wav2letter",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # build pte
    pte_filename = "wav2letter"
    instance = Wav2LetterModel()
    instance.vocab_size = 29
    model = instance.get_eager_model().eval()
    if args.weight:
        weight = torch.load(args.weight, weights_only=True)
        model.load_state_dict(weight)
    assert args.calibration_number
    if args.dataset:
        inputs = DataManager.get_dataset(
            data_dir=f"{args.dataset}",
            calinum=args.calibration_number,
        )
    else:
        inputs = [instance.get_example_inputs() for _ in range(args.calibration_number)]

    test_in = inputs[0]
    float_out = model(*test_in)

    compile_specs = [gen_samsung_backend_compile_spec(args.chipset)]

    if args.precision:
        model = quantize_module(
            model, inputs[0], inputs, getattr(Precision, args.precision)
        )
        quant_out = model(*test_in)

    edge_prog = to_edge_transform_and_lower_to_enn(
        model, inputs[0], compile_specs=compile_specs
    )

    edge = edge_prog.to_backend(EnnPartitioner(compile_specs))
    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    save_pte_program(exec_prog, pte_filename, os.path.join(f"{args.artifact}"))

    if args.dump:
        save_tensors(test_in, "float_in", args.artifact)
        save_tensors(float_out, "float_out", args.artifact)
        if args.precision:
            save_tensors(quant_out, "quant_out", args.artifact)
