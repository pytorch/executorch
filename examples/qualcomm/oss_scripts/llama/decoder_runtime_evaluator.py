# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, final, List, Optional, Union

import numpy as np
import torch
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    DECODER_MODEL_VERSION,
    EVAL_MODE,
    TEXT_DECODER,
    TEXT_EMBEDDING,
    VISION_ENCODER,
    VISION_ENCODER_INPUT_FILENAME,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    INFERENCE_REGISTRY,
    retrieve_info_from_pte,
)
from executorch.examples.qualcomm.utils import make_output_dir, SimpleADB
from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer
from torchao.quantization.utils import compute_error

try:
    from lm_eval.evaluator import simple_evaluate
except ImportError:
    raise ImportError(
        "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
    )


def post_process_model_output(output_holder: List, host_output_response_path: str):
    with open(host_output_response_path, "r") as f:
        output_holder.append(f.read())


def post_process_inference_speed(output_holder: List, host_performance_path: str):
    with open(host_performance_path, "r") as f:
        output_holder.append(float(f.read()))


def post_process_logits(
    output_holder: List,
    host_logits_path: str,
    kv_io_bit_width: int,
    num_input_tokens: int,
    output_vocab_size: int,
    logits_scale: float,
    logits_zero_point: int,
):
    with open(host_logits_path, "r") as f:
        logits_dtype = np.float32 if kv_io_bit_width == 32 else np.uint16
        output_tensor = torch.from_numpy(
            np.fromfile(f.name, dtype=logits_dtype).reshape(1, -1, output_vocab_size)
        )
        output_tensor = output_tensor[:, :num_input_tokens, :]
        output_tensor = (
            output_tensor.to(torch.float32) - logits_zero_point
        ) * logits_scale
        output_holder.append(output_tensor)


class EvalBase(ABC):
    _adb: Optional[SimpleADB] = None  # ADB shared across all instances

    def __init__(
        self,
        args: argparse.Namespace,
        pte_paths: Dict,
        runtime_tokenizer_path: str,
        is_modality: bool,
    ):
        self.args = args
        self.pte_paths = pte_paths
        self.runtime_tokenizer_path = runtime_tokenizer_path
        self.qnn_sdk = os.getenv("QNN_SDK_ROOT")
        self.is_modality = is_modality

        self.device_workspace = (
            f"/data/local/tmp/{getpass.getuser()}/executorch/static_llm"
        )
        self.runner = (
            "qnn_multimodal_runner" if self.is_modality else "qnn_llama_runner"
        )
        device_output_path = self._get_adb().output_folder
        if args.enable_x86_64:
            logging.warning(
                "x86 emulator is NOT recommended as it is for CI purpose, expect significance drop in performance."
            )
            device_output_path = f"{args.artifact}/outputs"

        self.device_output_response_path = f"{device_output_path}/outputs.txt"
        self.device_performance_path = f"{device_output_path}/inference_speed.txt"
        self.device_logits_path = f"{device_output_path}/all_logits.raw"
        self.host_output_response_path = f"{args.artifact}/outputs/outputs.txt"
        self.host_performance_path = f"{args.artifact}/outputs/inference_speed.txt"
        self.host_logits_path = f"{args.artifact}/outputs/all_logits.raw"
        make_output_dir(f"{args.artifact}/outputs")

        self.runner_base_cmd = self._init_runner_base_cmd()

    def _init_runner_base_cmd(self):
        """
        If a runner cmd could be shared across all EvalBase class, place it here to reduce redundant code.
        """
        args = self.args
        base_cmd = ""
        if args.enable_x86_64:
            base_cmd = " ".join(
                [
                    f"export LD_LIBRARY_PATH={self.qnn_sdk}/lib/x86_64-linux-clang/:{args.build_folder}/lib &&",
                    f"./{args.build_folder}/examples/qualcomm/oss_scripts/llama/{self.runner}",
                    f"--decoder_model_version {DECODER_MODEL_VERSION[args.decoder_model]}",
                    f"--tokenizer_path {self.runtime_tokenizer_path}",
                    f"--output_path {self.device_output_response_path}",
                    f"--performance_output_path {self.device_performance_path}",
                ]
            )
            if self.is_modality:
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--decoder_path {self.pte_paths[TEXT_DECODER]}",
                        f"--encoder_path {self.pte_paths[VISION_ENCODER]}",
                        f"--embedding_path {self.pte_paths[TEXT_EMBEDDING]}",
                        f"--image_path {args.artifact}/{VISION_ENCODER_INPUT_FILENAME}.raw",
                    ]
                )
            else:
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--model_path {self.pte_paths[TEXT_DECODER]}",
                    ]
                )
        else:
            base_cmd = " ".join(
                [
                    f"cd {self.device_workspace} &&",
                    f"./{self.runner}",
                    f"--decoder_model_version {DECODER_MODEL_VERSION[args.decoder_model]}",
                    f"--tokenizer_path {os.path.basename(self.runtime_tokenizer_path)}",
                    f"--output_path {self.device_output_response_path}",
                    f"--performance_output_path {self.device_performance_path}",
                    "--shared_buffer",
                ]
            )

            if self.is_modality:
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--decoder_path {os.path.basename(self.pte_paths[TEXT_DECODER])}",
                        f"--encoder_path {os.path.basename(self.pte_paths[VISION_ENCODER])}",
                        f"--embedding_path {os.path.basename(self.pte_paths[TEXT_EMBEDDING])}",
                        f"--image_path {VISION_ENCODER_INPUT_FILENAME}.raw",
                    ]
                )
            else:
                base_cmd = " ".join(
                    [
                        base_cmd,
                        f"--model_path {os.path.basename(self.pte_paths[TEXT_DECODER])}",
                    ]
                )

        return base_cmd

    @final
    def _get_adb(self):
        args = self.args
        if EvalBase._adb is None:
            EvalBase._adb = SimpleADB(
                qnn_sdk=self.qnn_sdk,
                build_path=f"{args.build_folder}",
                pte_path=list(self.pte_paths.values()),
                workspace=self.device_workspace,
                device_id=args.device,
                host_id=args.host,
                soc_model=args.model,
                runner=f"examples/qualcomm/oss_scripts/llama/{self.runner}",
                target=args.target,
            )
        return EvalBase._adb

    @abstractmethod
    def run(self) -> Any:
        # Performs execution and comparison.
        # Returns:
        #   Any: The result of execution.
        #   The type of the return value are determined by the implementation.
        #   The caller can return anything and handle it themselves.
        pass


class DefaultEval(EvalBase):
    def __init__(self, args, pte_paths, runtime_tokenizer_path, is_modality):
        super().__init__(args, pte_paths, runtime_tokenizer_path, is_modality)
        self.adb = self._get_adb()
        self.inference_speed = 0

        lookahead_args = " ".join(
            [
                f"--window {args.window}",
                f"--gcap {args.gcap}",
                f"--ngram {args.ngram}",
            ]
        )
        runner_args = " ".join(
            [
                f"--eval_mode {EVAL_MODE[args.model_mode]}",
                f"--temperature {args.temperature}",
                f"--system_prompt '{args.system_prompt}'",
                lookahead_args if args.model_mode == "lookahead" else "",
            ]
        )
        self.runner_cmd = " ".join(
            [
                self.runner_base_cmd,
                runner_args,
                f"--seq_len {args.max_seq_len}",
            ]
        )

    def run(self, prompt):
        multi_prompts = " ".join([f'--prompt "{p}"' for p in prompt])

        model_output_holder = []
        performance_holder = []

        if self.args.enable_x86_64:
            # x86 emulator is intended for CI and not performance. Check only the first few tokens.
            seq_len = min(self.args.max_seq_len, 32)
            runner_cmd = " ".join(
                [
                    self.runner_cmd,
                    multi_prompts,
                    f"--seq_len {seq_len}",
                ]
            )
            subprocess.run(
                runner_cmd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
            )
            post_process_model_output(
                output_holder=model_output_holder,
                host_output_response_path=self.host_output_response_path,
            )
            post_process_inference_speed(
                output_holder=performance_holder,
                host_performance_path=self.host_performance_path,
            )
        else:
            runner_cmd = " ".join(
                [self.runner_cmd, multi_prompts, f"--seq_len {self.args.max_seq_len}"]
            )

            extra_files = [self.runtime_tokenizer_path]
            if self.is_modality:
                extra_files = extra_files + [
                    f"{self.args.artifact}/{VISION_ENCODER_INPUT_FILENAME}.raw"
                ]
            self.adb.push(inputs=[], files=extra_files)
            self.adb.execute(custom_runner_cmd=runner_cmd)
            self.adb.pull(
                host_output_path=self.host_output_response_path,
                device_output_path=self.device_output_response_path,
                callback=partial(
                    post_process_model_output,
                    output_holder=model_output_holder,
                    host_output_response_path=self.host_output_response_path,
                ),
            )
            self.adb.pull(
                host_output_path=self.host_performance_path,
                device_output_path=self.device_performance_path,
                callback=partial(
                    post_process_inference_speed,
                    output_holder=performance_holder,
                    host_performance_path=self.host_performance_path,
                ),
            )
        self.inference_speed = performance_holder[0]
        return model_output_holder


class SqnrEval(EvalBase):
    """
    SQNR Evaluator will only evaluate the given prompt's logit output.
    It won't evaluate the generated token's logit.
    """

    def __init__(
        self,
        source_model,
        get_example_inputs: Callable,
        args,
        pte_paths,
        tokenizer,
        runtime_tokenizer_path,
        is_modality,
    ):
        super().__init__(args, pte_paths, runtime_tokenizer_path, is_modality)
        self.inference_speed = 0
        self.source_model = source_model
        self.get_example_inputs = get_example_inputs
        self.adb = self._get_adb()
        self.tokenizer = tokenizer
        self.enable_x86_64 = args.enable_x86_64
        self.max_seq_length = args.max_seq_len

        pte_meta_info = retrieve_info_from_pte(pte_path=pte_paths[TEXT_DECODER])
        self.output_vocab_size = pte_meta_info["output_vocab_size"]
        pte_max_seq_len = pte_meta_info["pte_max_seq_len"]
        self.logits_scale = pte_meta_info["logits_scale"]
        self.logits_zero_point = pte_meta_info["logits_zero_point"]
        self.kv_io_bit_width = pte_meta_info["kv_io_bit_width"]

        if args.model_mode != "kv":
            logging.warning(
                f"Current Sqnr Eval does not support {args.model_mode}, switching to kv mode."
            )

        if pte_max_seq_len != self.max_seq_length:
            logging.warning(
                f"The pte provided has a max_seq_len {pte_max_seq_len}, which is different from --max_seq_len {self.max_seq_length} provided to the script, please ensure this is desired."
            )
            if pte_max_seq_len < self.max_seq_length:
                logging.warning(
                    f"The pte max_seq_len {pte_max_seq_len} is used since it is shorter than --max_seq_len {self.max_seq_length}"
                )
                self.max_seq_length = pte_max_seq_len

    def run(self, prompt):
        golden_logits = INFERENCE_REGISTRY[True](
            get_example_inputs=self.get_example_inputs,
            prompt=prompt,
            module=self.source_model,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_length,
            use_i64_token=self.args.embedding_quantize is not None,
            collect_logits=True,
        )

        input_file_name = f"{self.args.artifact}/input_tokens.raw"

        # Make sure the encode param aligns with encode param under all register_inference function.
        # This ensures the token used for device is same as token used for nn.Module.
        inps = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False))
        inps = inps.to(torch.uint64).numpy()
        inps.tofile(input_file_name)

        assert (
            inps.size < self.max_seq_length
        ), f"Number of input token is longer than max_seq_len, please shorten the input token length. input_token length: {inps.size}. max_seq_len: {self.max_seq_length}"

        output_logits_holder = []
        output_performance_holder = []

        if self.enable_x86_64:
            runner_cmd = " ".join(
                [
                    self.runner_base_cmd,
                    f"--seq_len {inps.size + 1}",
                    f"--eval_mode {EVAL_MODE['kv']}",
                    "--temperature 0",
                    f"--dump_logits_path {self.device_logits_path}",
                    f"--tokenized_prompt {input_file_name}",
                ]
            )

            subprocess.run(
                runner_cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            post_process_logits(
                output_holder=output_logits_holder,
                host_logits_path=self.host_logits_path,
                kv_io_bit_width=self.kv_io_bit_width,
                num_input_tokens=inps.size,
                output_vocab_size=self.output_vocab_size,
                logits_scale=self.logits_scale,
                logits_zero_point=self.logits_zero_point,
            )
            post_process_inference_speed(
                output_holder=output_performance_holder,
                host_performance_path=self.host_performance_path,
            )
        else:
            runner_cmd = " ".join(
                [
                    self.runner_base_cmd,
                    f"--seq_len {inps.size + 1}",
                    f"--eval_mode {EVAL_MODE['kv']}",
                    "--temperature 0",
                    f"--dump_logits_path {self.device_logits_path}",
                    f"--tokenized_prompt {os.path.basename(input_file_name)}",
                ]
            )
            self.adb.push(
                inputs=[], files=[input_file_name, self.runtime_tokenizer_path]
            )
            self.adb.execute(custom_runner_cmd=runner_cmd)
            self.adb.pull(
                host_output_path=self.host_logits_path,
                device_output_path=self.device_logits_path,
                callback=partial(
                    post_process_logits,
                    output_holder=output_logits_holder,
                    host_logits_path=self.host_logits_path,
                    kv_io_bit_width=self.kv_io_bit_width,
                    num_input_tokens=inps.size,
                    output_vocab_size=self.output_vocab_size,
                    logits_scale=self.logits_scale,
                    logits_zero_point=self.logits_zero_point,
                ),
            )
            self.adb.pull(
                host_output_path=self.host_performance_path,
                device_output_path=self.device_performance_path,
                callback=partial(
                    post_process_inference_speed,
                    output_holder=output_performance_holder,
                    host_performance_path=self.host_performance_path,
                ),
            )
        self.inference_speed = output_performance_holder[0]

        sqnr = compute_error(golden_logits, output_logits_holder[0])
        return sqnr.item(), golden_logits, output_logits_holder[0]


class TaskEval(EvalBase):
    class QnnRunnerEvalWrapper(EagerEvalWrapper):
        """
        A wrapper class to run tasks with QNN on device.
        """

        def __init__(  # noqa: C901
            self,
            args,
            runner_base_cmd: str,
            adb: SimpleADB,
            pte_path: str,
            device_performance_path: str,
            device_logits_path: str,
            host_performance_path: str,
            host_logits_path: str,
            tokenizer: Union[
                SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer
            ],
            runtime_tokenizer_path,
        ):
            self.inference_speed = None

            self.args = args
            self.runner_base_cmd = runner_base_cmd
            self.adb = adb
            self.pte_path = pte_path
            self.runtime_tokenizer_path = runtime_tokenizer_path

            self.device_performance_path = device_performance_path
            self.device_logits_path = device_logits_path
            self.host_performance_path = host_performance_path
            self.host_logits_path = host_logits_path

            self.enable_x86_64 = args.enable_x86_64
            self.max_seq_length = args.max_seq_len
            pte_meta_info = retrieve_info_from_pte(pte_path=self.pte_path)
            self.output_vocab_size = pte_meta_info["output_vocab_size"]
            pte_max_seq_len = pte_meta_info["pte_max_seq_len"]
            self.logits_scale = pte_meta_info["logits_scale"]
            self.logits_zero_point = pte_meta_info["logits_zero_point"]
            self.kv_io_bit_width = pte_meta_info["kv_io_bit_width"]

            if args.model_mode != "kv":
                logging.warning(
                    f"Current QnnRunnerEvalWrapper does not support {args.model_mode}, switching to kv mode."
                )

            if pte_max_seq_len != self.max_seq_length:
                logging.warning(
                    f"The pte provided has a max_seq_len {pte_max_seq_len}, which is different from --max_seq_len {self.max_seq_length} provided to the script, please ensure this is desired."
                )
                if pte_max_seq_len < self.max_seq_length:
                    logging.warning(
                        f"The pte max_seq_len {pte_max_seq_len} is used since it is shorter than --max_seq_len {self.max_seq_length}"
                    )
                    self.max_seq_length = pte_max_seq_len

            if not self.enable_x86_64:
                self.adb.push(inputs=[], files=[self.runtime_tokenizer_path])
            # n seq len = n-1 cache len, so we set len(inps) = n-1 during _model_call
            # pyre-ignore
            super().__init__(None, tokenizer, self.max_seq_length - 1)

        def _model_call(self, inps):
            input_file_name = f"{self.args.artifact}/input_tokens.raw"
            # This is the dtype required by runtime tokenizer.
            inps = inps.to(torch.uint64).numpy()
            inps.tofile(input_file_name)
            output_logits_holder = []
            output_performance_holder = []

            if self.enable_x86_64:
                runner_cmd = " ".join(
                    [
                        self.runner_base_cmd,
                        f"--seq_len {self.max_seq_length}",
                        f"--eval_mode {EVAL_MODE['kv']}",
                        "--temperature 0",
                        f"--dump_logits_path {self.device_logits_path}",
                        f"--tokenized_prompt {input_file_name}",
                    ]
                )
                subprocess.run(
                    runner_cmd,
                    shell=True,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                post_process_logits(
                    output_holder=output_logits_holder,
                    host_logits_path=self.host_logits_path,
                    kv_io_bit_width=self.kv_io_bit_width,
                    num_input_tokens=inps.size,
                    output_vocab_size=self.output_vocab_size,
                    logits_scale=self.logits_scale,
                    logits_zero_point=self.logits_zero_point,
                )
                post_process_inference_speed(
                    output_holder=output_performance_holder,
                    host_performance_path=self.host_performance_path,
                )

            else:
                runner_cmd = " ".join(
                    [
                        self.runner_base_cmd,
                        f"--seq_len {self.max_seq_length}",
                        f"--eval_mode {EVAL_MODE['kv']}",
                        "--temperature 0",
                        f"--dump_logits_path {self.device_logits_path}",
                        f"--tokenized_prompt {os.path.basename(input_file_name)}",
                    ]
                )
                self.adb.push(inputs=[], files=[input_file_name], init_env=False)
                self.adb.execute(custom_runner_cmd=runner_cmd)
                self.adb.pull(
                    host_output_path=self.host_logits_path,
                    device_output_path=self.device_logits_path,
                    callback=partial(
                        post_process_logits,
                        output_holder=output_logits_holder,
                        host_logits_path=self.host_logits_path,
                        kv_io_bit_width=self.kv_io_bit_width,
                        num_input_tokens=inps.size,
                        output_vocab_size=self.output_vocab_size,
                        logits_scale=self.logits_scale,
                        logits_zero_point=self.logits_zero_point,
                    ),
                )
                self.adb.pull(
                    host_output_path=self.host_performance_path,
                    device_output_path=self.device_performance_path,
                    callback=partial(
                        post_process_inference_speed,
                        output_holder=output_performance_holder,
                        host_performance_path=self.host_performance_path,
                    ),
                )
            self.inference_speed = output_performance_holder[0]
            return output_logits_holder[0]

    def __init__(self, args, pte_paths, tokenizer, runtime_tokenizer_path, is_modality):
        super().__init__(
            args=args,
            pte_paths=pte_paths,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_modality=is_modality,
        )
        self.inference_speed = None
        self.tasks = args.tasks
        self.num_fewshot = args.num_fewshot
        self.limit = args.limit
        adb = self._get_adb()
        self.eval_wrapper = TaskEval.QnnRunnerEvalWrapper(
            args=args,
            runner_base_cmd=self.runner_base_cmd,
            adb=adb,
            pte_path=self.pte_paths[TEXT_DECODER],
            device_performance_path=self.device_performance_path,
            device_logits_path=self.device_logits_path,
            host_performance_path=self.host_performance_path,
            host_logits_path=self.host_logits_path,
            tokenizer=tokenizer,
            runtime_tokenizer_path=self.runtime_tokenizer_path,
        )

    def run(self):
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=self.eval_wrapper,
                tasks=self.tasks,
                num_fewshot=self.num_fewshot,
                limit=self.limit,
            )
        self.inference_speed = self.eval_wrapper.inference_speed
        return eval_results
