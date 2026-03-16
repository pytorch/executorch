# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import os

from typing import Optional, Union

import numpy as np

import torch
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper

from executorch.examples.qualcomm.utils import (
    make_output_dir,
    setup_common_args_and_variables,
)

from lm_eval.evaluator import simple_evaluate
from pytorch_tokenizers import get_tokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer as Tiktoken


def create_device_inputs(example_inputs):
    # TODO: support batch inputs if necessary
    input_list = ""
    inputs = []

    for index, data in enumerate(example_inputs):
        inputs.append(data)
        input_list += " ".join([f"input_{index}_{i}.raw" for i in range(len(data))])
        input_list += "\n"
    return inputs, input_list


class GraphModuleCalibrationWrapper(EagerEvalWrapper):
    """
    A wrapper class for calibration
    """

    def __init__(
        self,
        model: torch.fx.GraphModule,
        tokenizer: Union[SentencePieceTokenizer, Tiktoken],
        max_seq_length: Optional[int] = None,
        use_kv_cache: bool = False,
        generate_full_logits: bool = False,
        enable_dynamic_shape: bool = True,
    ):
        super().__init__(
            model=model, tokenizer=tokenizer, max_seq_length=max_seq_length
        )
        self._model = model.to(self.device)
        self._use_kv_cache = use_kv_cache
        self._generate_full_logits = generate_full_logits
        self._enable_dynamic_shape = enable_dynamic_shape

    def _model_call(self, inps):
        if self._use_kv_cache:
            if not self._enable_dynamic_shape:
                # graph module exported without dynamic shape won't work with a different shape.
                # And we have to do single token prefill here.
                result_logits = []
                for pos in range(inps.shape[-1]):
                    pos_tensor = torch.tensor([pos], dtype=torch.int64)
                    logits = self._model(inps[:, pos : pos + 1], pos_tensor)
                    result_logits.append(logits)
                if self._generate_full_logits:
                    return torch.cat(result_logits, dim=1)
                else:
                    return torch.stack(result_logits, dim=1)
            else:
                pos_tensor = torch.tensor([0], dtype=torch.int64, device=self.device)
                # Batch process the whole sequence.
                logits = self._model(inps[:, : self._max_seq_length], pos_tensor)
                return logits

        else:
            return self._model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


class QNNRunnerEvalWrapper(EagerEvalWrapper):
    """
    A wrapper class for ExecuTorch Runtime integration with the
    lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Union[SentencePieceTokenizer, Tiktoken],
        soc_model: str,
        device: str,
        host: str,
        max_seq_length: Optional[int] = None,
        output_dir: str = ".",
        quant_attrs=None,
        build_folder: str = "build-android",
        target="aarch64-android",
    ):
        super().__init__(None, tokenizer, max_seq_length)
        import getpass

        from executorch.examples.qualcomm.utils import SimpleADB

        self._model = model
        self.output_dir = output_dir
        self.quant_attrs = quant_attrs
        workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/meta_llama"
        self.adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=build_folder,
            pte_path=model,
            workspace=workspace,
            device_id=device,
            host_id=host,
            soc_model=soc_model,
            target=target,
        )
        self.adb.push()

    def _model_call(self, inps):
        # Given inps (tokens), return the logits from a single
        # forward call

        # Example:
        # inps: Tensor of shape (1, N)
        # logits: Tensor of shape (1, N, vocab_size)
        result_logits = []
        inputs = []

        for pos in range(self._max_seq_length):
            pos_tensor = torch.tensor([pos], dtype=torch.int64)
            inputs.append([inps[:, pos : pos + 1], pos_tensor])

        inputs, input_list = create_device_inputs(inputs)
        self.adb.push(inputs=inputs, input_list=input_list, init_env=False)
        self.adb.execute()
        output_data_folder = f"{self.output_dir}/outputs"
        make_output_dir(output_data_folder)

        def post_process():
            for f in sorted(
                os.listdir(output_data_folder), key=lambda f: int(f.split("_")[1])
            ):
                output_tensor = None
                if self.quant_attrs:
                    output_tensor = torch.from_numpy(
                        np.fromfile(
                            os.path.join(output_data_folder, f), dtype=np.uint16
                        ).reshape(1, 1, -1)
                    )
                    output_tensor = (
                        output_tensor.to(torch.float32) - self.quant_attrs["zero_point"]
                    ) * self.quant_attrs["scale"]
                else:
                    output_tensor = torch.from_numpy(
                        np.fromfile(
                            os.path.join(output_data_folder, f), dtype=np.float32
                        ).reshape(1, 1, -1)
                    )

                result_logits.append(output_tensor)

        self.adb.pull(host_output_path=self.output_dir, callback=post_process)
        return torch.cat(result_logits, dim=1)


def gen_eval_wrapper(
    args: argparse.ArgumentParser,
):
    """
    Generates a wrapper interface around the provided model and tokenizer for
    the lm-evaluation-harness library.

    Returns:
        eval_wrapper (LM): A wrapper interface for the lm-evaluation-harness library.
    """
    tokenizer = get_tokenizer(args.tokenizer_path)

    # ExecuTorch Binary Evaluation
    if (model := args.pte) is not None:  # pyre-ignore
        assert args.device is not None, "please specify the device to execute pte"
        quant_attrs = None
        if args.logits_quant_attr_path is not None:
            quant_attrs = json.load(open(f"{args.logits_quant_attr_path}"))
        return QNNRunnerEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            soc_model=args.model,
            device=args.device,
            host=args.host,
            max_seq_length=args.max_seq_len - 1,
            output_dir=args.artifact,
            quant_attrs=quant_attrs,
            build_folder=args.build_folder,
            target=args.target,
        )
    else:
        raise RuntimeError("Currently only support evaluate pte on device")


def eval_llama(
    args: argparse.ArgumentParser,
) -> None:

    # Generate the eval wrapper
    eval_wrapper = gen_eval_wrapper(args)

    # Evaluate the model
    with torch.no_grad():
        eval_results = simple_evaluate(
            model=eval_wrapper,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )

    for task, res in eval_results["results"].items():
        print(f"{task}: {res}")


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example.",
        type=str,
    )

    parser.add_argument(
        "--tokenizer_path",
        help="path to tokenizer.json.",
        type=str,
    )

    parser.add_argument(
        "--pte",
        type=str,
        default=None,
        help="[For ExecuTorch] Path to the ExecuTorch model being evaluated. If provided, don't go through the export flow",
    )
    parser.add_argument(
        "--logits_quant_attr_path",
        type=str,
        default=None,
        help="For the pte with tag quant io, it needs to be dequantize and compute ppl.",
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="This refers to maximum number of tokens that the model can process & consider at once to generate predictions/responses.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="list of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="number of samples to evalulate. If not set, evaluate all samples",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )

    args = parser.parse_args()

    eval_llama(args)


if __name__ == "__main__":
    main()  # pragma: no cover
