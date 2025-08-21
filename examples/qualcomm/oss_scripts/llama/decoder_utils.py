# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import logging
import os
from typing import Callable, Optional, Union

import numpy as np

import torch
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper

from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    DECODER_MODEL_VERSION,
    EVAL_MODE,
)
from executorch.examples.qualcomm.utils import make_output_dir, SimpleADB
from executorch.exir._serialize._program import deserialize_pte_binary
from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer

try:
    from lm_eval.evaluator import simple_evaluate
except ImportError:
    raise ImportError(
        "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
    )


class GraphModuleCalibrationWrapper(EagerEvalWrapper):
    """
    A wrapper class for calibration
    """

    def __init__(
        self,
        model: torch.fx.GraphModule,
        tokenizer: Union[
            SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer
        ],
        max_seq_length: int,
        ar_len: int,
        use_kv_cache: bool,
        get_example_inputs: Callable,
        kv_updater: Callable,
        use_i64_token: bool,
    ):
        # n seq len = n-1 cache len, so we len(inps) = n-1 during _model_call
        assert max_seq_length is not None, "max_seq_length must be provided"
        super().__init__(
            model=model, tokenizer=tokenizer, max_seq_length=max_seq_length - 1
        )
        self._model = model.to(self.device)
        self.ar_len = ar_len
        self._use_kv_cache = use_kv_cache
        self.get_example_inputs = get_example_inputs
        self.max_seq_length = max_seq_length
        self.kv_updater = kv_updater
        self.use_i64_token = use_i64_token

    def _model_call(self, inps):
        all_logits = None
        if self._use_kv_cache:
            all_logits = kv_inference(
                self.get_example_inputs,
                inps,
                self._model,
                self._tokenizer,
                self.ar_len,
                self.max_seq_length,
                kv_updater=self.kv_updater,
                use_i64_token=self.use_i64_token,
                collect_logits=True,
            )
        else:
            all_logits = prefill_inference(
                self.get_example_inputs,
                inps,
                self._model,
                self._tokenizer,
                self.ar_len,
                self.max_seq_length,
                use_i64_token=self.use_i64_token,
                collect_logits=True,
            )
        return all_logits


class QnnRunnerEvalWrapper(EagerEvalWrapper):
    """
    A wrapper class to run PPL scores with QNN on device.
    """

    def __init__(
        self,
        args,
        pte_path: str,
        tokenizer: Union[
            SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer
        ],
        runtime_tokenizer_path,
        max_seq_length: int,
    ):
        self.args = args
        self.pte_path = pte_path

        with open(pte_path, "rb") as f:
            program_data = f.read()
        program = deserialize_pte_binary(program_data)

        # Retrieve vocab_size from get_metadata under static_llama that is passed to edge manager
        self.output_vocab_size = None
        pte_max_seq_len = None
        self.logits_scale = None
        self.logits_zero_point = None
        self.kv_io_bit_width = 32
        for method in program.execution_plan:
            # Don't use tokenizer.n_words, the numbers are off once calling get_tokenizer()
            if method.name == "get_vocab_size":
                # pyre-ignore
                self.output_vocab_size = method.values[0].val.int_val
            if method.name == "get_max_seq_len":
                # pyre-ignore
                pte_max_seq_len = method.values[0].val.int_val
            if method.name == "get_logits_scale":
                self.logits_scale = method.values[0].val.double_val
            if method.name == "get_logits_zero_point":
                self.logits_zero_point = method.values[0].val.int_val
            if method.name == "get_kv_io_bit_width":
                self.kv_io_bit_width = method.values[0].val.int_val

        # FP has no scale/zero_point, use following values, which is equivalent to not performing dequantize.
        if self.kv_io_bit_width == 32:
            self.logits_scale = 1
            self.logits_zero_point = 0
        elif self.logits_scale is None or self.logits_zero_point is None:
            raise RuntimeError(
                "Unable to find scale/offset. The .pte file might be deprecated. Please generate a new .pte file"
            )

        assert self.output_vocab_size is not None, "Couldn't find the vocab size"
        assert pte_max_seq_len is not None, "Couldn't find the max_seq_len from pte"
        if pte_max_seq_len != max_seq_length:
            logging.warning(
                f"The pte provided has a max_seq_len {pte_max_seq_len}, which is different from --max_seq_len {max_seq_length} provided to the script, please ensure this is desired."
            )
            if pte_max_seq_len < max_seq_length:
                logging.warning(
                    f"The pte max_seq_len {pte_max_seq_len} is used since it is shorter than --max_seq_len {max_seq_length}"
                )
                max_seq_length = pte_max_seq_len
        self.max_seq_length = max_seq_length
        self.runtime_tokenizer_path = runtime_tokenizer_path

        self.output_dir = args.artifact

        self.workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"
        self.adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=args.build_folder,
            pte_path=pte_path,
            workspace=self.workspace,
            device_id=args.device,
            host_id=args.host,
            soc_model=args.model,
            runner="examples/qualcomm/oss_scripts/llama/qnn_llama_runner",
        )
        self.adb.push(inputs=[], files=[self.runtime_tokenizer_path])
        # n seq len = n-1 cache len, so we len(inps) = n-1 during _model_call
        # pyre-ignore
        super().__init__(None, tokenizer, max_seq_length - 1)

    def _model_call(self, inps):

        input_file_name = f"{self.args.artifact}/input_tokens.raw"
        inps = inps.to(torch.uint64).numpy()
        inps.tofile(input_file_name)

        outputs_path = "outputs/outputs.txt"
        dump_logits_path = "outputs/all_logit.raw"
        performance_output_path = "outputs/inference_speed.txt"
        runner_cmd = " ".join(
            [
                f"cd {self.workspace} &&",
                "./qnn_llama_runner",
                f"--decoder_model_version {DECODER_MODEL_VERSION[self.args.decoder_model]}",
                f"--tokenizer_path {os.path.basename(self.runtime_tokenizer_path)}",
                f"--model_path {os.path.basename(self.pte_path)}",
                f"--seq_len {self.max_seq_length}",
                f"--output_path {outputs_path}",
                f"--performance_output_path {performance_output_path}",
                f"--kv_updater {'SmartMask' if self.args.kv_updater == smart_mask_updater else 'ShiftPointer'}",
                f"--window {self.args.window}",
                f"--gcap {self.args.gcap}",
                f"--ngram {self.args.ngram}",
                f"--eval_mode {EVAL_MODE[self.args.model_mode]}",
                "--temperature 0",
                f"--dump_logits_path {dump_logits_path}",
                f"--tokenized_prompt {os.path.basename(input_file_name)}",
            ]
        )

        self.adb.push(inputs=[], files=[input_file_name], init_env=False)
        self.adb.execute(custom_runner_cmd=runner_cmd)
        output_data_folder = f"{self.output_dir}/outputs"
        make_output_dir(output_data_folder)
        output_tensor_list = []

        def post_process():
            with open(f"{self.args.artifact}/{dump_logits_path}", "r") as f:
                output_tensor = torch.from_numpy(
                    np.fromfile(f.name, dtype=np.uint16).reshape(
                        1, -1, self.output_vocab_size
                    )
                )
                output_tensor = (
                    output_tensor.to(torch.float32) - self.logits_zero_point
                ) * self.logits_scale
                output_tensor_list.append(output_tensor)

            # simple_eval will run multiple rounds, use last run for inference speed
            with open(f"{self.args.artifact}/{performance_output_path}", "r") as f:
                self.inference_speed = float(f.read())

        self.adb.pull(output_path=self.output_dir, callback=post_process)
        return output_tensor_list[0]


def smart_mask_updater(
    _, n_updates, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
):
    # ar_len is unused in smart mask
    max_cache_len = k_caches[0].size(-1)
    if pos + n_updates <= max_cache_len:
        for i, k_cache in enumerate(k_caches):
            k_cache[:, :, pos : pos + n_updates] = new_k_caches[i][:, :, :n_updates]

        for i, v_cache in enumerate(v_caches):
            v_cache[:, pos : pos + n_updates, :] = new_v_caches[i][:, :n_updates, :]
        atten_mask[:, :, pos : pos + n_updates] = 0
    pos += n_updates

    return (atten_mask, pos, k_caches, v_caches)


def shift_pointer_updater(
    ar_len, n_updates, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
):
    max_cache_len = k_caches[0].size(-1)
    if pos + n_updates <= max_cache_len:
        k_caches = [
            torch.cat(
                [k_cache[:, :, n_updates:], new_k_caches[i][:, :, :n_updates]], dim=-1
            )
            for i, k_cache in enumerate(k_caches)
        ]
        v_caches = [
            torch.cat(
                [v_cache[:, n_updates:, :], new_v_caches[i][:, :n_updates, :]], dim=1
            )
            for i, v_cache in enumerate(v_caches)
        ]
        atten_mask[:, :, -pos - n_updates - ar_len : -pos - ar_len] = 0
    pos += n_updates

    return (atten_mask, pos, k_caches, v_caches)


def kv_inference(
    get_example_inputs,
    prompt: Union[str, list],
    module: torch.fx.GraphModule,
    tokenizer,
    ar_len=1,
    max_seq_len=512,
    kv_updater=smart_mask_updater,
    use_i64_token=False,
    collect_logits=False,
):
    _, atten_mask, _, k_caches, v_caches = get_example_inputs(use_kv_cache=True)

    # TODO: change criteria & support batch inputs if necessary
    all_pos = torch.arange(0, max_seq_len, 1, dtype=torch.int32).unsqueeze(0)

    prompt_token_list, total_token_list, result_logits = [], [], []

    if isinstance(prompt, str):
        # Llama2 tokenizer has no special tokens
        if isinstance(tokenizer, (SentencePieceTokenizer, HuggingFaceTokenizer)):
            prompt_token_list = tokenizer.encode(prompt, bos=True, eos=False)
        elif isinstance(tokenizer, TiktokenTokenizer):
            prompt_token_list = tokenizer.encode(
                prompt, bos=True, eos=False, allowed_special="all"
            )
        else:
            raise RuntimeError("Unknown tokenizer")
    else:
        # pyre-ignore
        prompt_token_list = prompt.flatten().tolist()
    total_token_list = prompt_token_list
    dtype = torch.int64 if use_i64_token else torch.int32

    with torch.no_grad():
        # Phase 1: Prefill the prompt in ar_len chunks.
        num_prompt_tokens = len(prompt_token_list)
        pos = 0  # Tracks how many prompt tokens have been processed.
        while pos < num_prompt_tokens:
            chunk_start_idx = pos
            # Take a chunk of prompt tokens, up to ar_len length.
            chunk_end_idx = min(num_prompt_tokens, pos + ar_len)
            actual_chunk_tokens = prompt_token_list[chunk_start_idx:chunk_end_idx]
            num_tokens_in_chunk = len(actual_chunk_tokens)

            # Prepare tmp_token_list (padded with zeros).
            tmp_token_list = torch.zeros((1, ar_len), dtype=dtype)
            tmp_token_list[0, :num_tokens_in_chunk] = torch.tensor(
                actual_chunk_tokens, dtype=dtype
            )

            # Prepare tmp_pos (padded with zeros).
            tmp_pos = torch.zeros((1, ar_len), dtype=torch.int32)
            tmp_pos[0, :num_tokens_in_chunk] = all_pos[
                0,
                pos : pos + num_tokens_in_chunk,
            ]

            # Run inference.
            logits, new_k_caches, new_v_caches = module(
                tmp_token_list,
                atten_mask,
                tmp_pos,
                *k_caches,
                *v_caches,
            )
            if collect_logits:
                result_logits.append(logits[:, :num_tokens_in_chunk])

            # Update the pos, KV cache and attention mask.
            atten_mask, pos, k_caches, v_caches = kv_updater(
                ar_len,
                num_tokens_in_chunk,
                atten_mask,
                pos,
                k_caches,
                v_caches,
                new_k_caches,
                new_v_caches,
            )
        # Append the last run logits to the total_token_list.
        total_token_list.append(
            torch.argmax(logits[:, num_tokens_in_chunk - 1], dim=-1).item()
        )

        # Phase 2: Generate tokens until the EOS token is generated or max_seq_len is reached.
        # When run on wikitext for ppl evaluation, this while-loop is not expected to run.
        max_cache_len = max_seq_len - ar_len
        num_tokens = len(total_token_list)
        while total_token_list[-1] != tokenizer.eos_id and num_tokens < max_seq_len:
            chunk_start_idx = min(pos, max_cache_len)
            # Take a chunk of generated tokens, up to ar_len length.
            chunk_end_idx = num_tokens
            actual_chunk_tokens = total_token_list[chunk_start_idx:chunk_end_idx]
            num_tokens_in_chunk = len(actual_chunk_tokens)

            # Prepare tmp_token_list (padded with zeros).
            tmp_token_list = torch.zeros((1, ar_len), dtype=dtype)
            tmp_token_list[0, :num_tokens_in_chunk] = torch.tensor(
                actual_chunk_tokens, dtype=dtype
            )

            # Prepare tmp_pos (padded with zeros).
            tmp_pos = torch.zeros((1, ar_len), dtype=torch.int32)
            tmp_pos[0, :num_tokens_in_chunk] = all_pos[0, chunk_start_idx:chunk_end_idx]

            logits, new_k_caches, new_v_caches = module(
                tmp_token_list,
                atten_mask,
                tmp_pos,
                *k_caches,
                *v_caches,
            )
            if collect_logits:
                result_logits.append(logits[:, :num_tokens_in_chunk])

            atten_mask, pos, k_caches, v_caches = kv_updater(
                ar_len,
                1,
                atten_mask,
                pos,
                k_caches,
                v_caches,
                new_k_caches,
                new_v_caches,
            )
            total_token_list.append(
                torch.argmax(logits[:, num_tokens_in_chunk - 1], dim=-1).item()
            )
            num_tokens = len(total_token_list)
    logging.info(f"kv inference result:\n{tokenizer.decode(total_token_list)}")
    if collect_logits:
        result_logits = torch.cat(result_logits, dim=1)
    return result_logits


def prefill_inference(
    get_example_inputs,
    prompt: Union[str, list],
    module: torch.fx.GraphModule,
    tokenizer,
    max_seq_len=512,
    use_i64_token=False,
    collect_logits=False,
):
    _, atten_mask = get_example_inputs(use_kv_cache=False)

    # TODO: change criteria & support batch inputs if necessary

    token_list, result_logits = [], []

    if isinstance(prompt, str):
        # Llama2 tokenizer has no special tokens
        if isinstance(tokenizer, (SentencePieceTokenizer, HuggingFaceTokenizer)):
            token_list = tokenizer.encode(prompt, bos=True, eos=False)
        elif isinstance(tokenizer, TiktokenTokenizer):
            token_list = tokenizer.encode(
                prompt, bos=True, eos=False, allowed_special="all"
            )
        else:
            raise RuntimeError("Unknown tokenizer")
    else:
        # pyre-ignore
        token_list = prompt.flatten().tolist()

    pos = len(token_list)
    dtype = torch.int64 if use_i64_token else torch.int32

    with torch.no_grad():
        while token_list[-1] != tokenizer.eos_id and pos < max_seq_len:
            tmp_token_list = torch.tensor(token_list, dtype=dtype).reshape(1, -1)
            if pos < max_seq_len:
                tmp_token_list = torch.cat(
                    [
                        tmp_token_list,
                        torch.zeros((1, max_seq_len - pos), dtype=dtype),
                    ],
                    dim=1,
                )
            results = module(
                tmp_token_list,
                atten_mask,
            )
            if len(results) == 3:
                logits, new_k_caches, new_v_caches = results
            elif len(results) == 1:
                logits = results
            logits = torch.argmax(logits[:, pos - 1], dim=-1).item()
            token_list.append(logits)
            if collect_logits:
                result_logits.append(logits)
            pos += 1

    logging.info(f"prefill inference result:\n{tokenizer.decode(token_list)}")
    if collect_logits:
        result_logits = torch.cat(result_logits, dim=1)
    return result_logits


def graph_module_inference(
    use_kv_cache: bool,
    get_example_inputs: Callable,
    module: torch.fx.GraphModule,
    tokenizer,
    ar_len=1,
    max_seq_len=512,
    kv_updater=smart_mask_updater,
    prompt=None,
    tasks=None,
    tasks_limit=1,
    num_fewshot=None,
    use_i64_token=False,
    event_name: Optional[str] = None,
):
    """
    This function supports model execution from static nn.Module decoder model
    all the way to edge program.
    Users could choose to provide either the prompt or tasks for execution but not both.
    """
    # Checks 1 and only 1 is provided.
    assert (tasks is None) != (
        prompt is None
    ), "Please provide either tasks or prompt - not both or neither"
    if tasks is None:
        if use_kv_cache:
            kv_inference(
                get_example_inputs,
                prompt,
                module,
                tokenizer,
                ar_len,
                max_seq_len,
                kv_updater=kv_updater,
                use_i64_token=use_i64_token,
                collect_logits=False,
            )
        else:
            prefill_inference(
                get_example_inputs,
                prompt,
                module,
                tokenizer,
                max_seq_len,
                use_i64_token,
                collect_logits=False,
            )
    else:
        calibration_wrapper = GraphModuleCalibrationWrapper(
            model=module,
            tokenizer=tokenizer,
            max_seq_length=max_seq_len,
            ar_len=ar_len,
            use_kv_cache=use_kv_cache,
            get_example_inputs=get_example_inputs,
            kv_updater=kv_updater,
            use_i64_token=use_i64_token,
        )
        # Evaluate the model
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=calibration_wrapper,
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=tasks_limit,
            )
        logging.info(f"Perplexity evaluation summary for {event_name}")
        for task, res in eval_results["results"].items():
            logging.info(f"{task}: {res}")


def apply_prompt_template(
    chat_template: Callable, prompt: str, system_prompt: str = None
):
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    template_prompt = chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logging.info(f"Prompt after applying template: {template_prompt}")
    return template_prompt
