# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import logging
import os
import subprocess
from collections import defaultdict, OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from executorch.backends.qualcomm._passes import SeqMSE
from executorch.examples.models.llama.evaluate.eager_eval import EagerEvalWrapper
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    DECODER_MODEL_VERSION,
    EVAL_MODE,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask

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


INFERENCE_REGISTRY = {}


def register_inference(use_kv_cache: bool):
    def decorator(func):
        INFERENCE_REGISTRY[use_kv_cache] = func

    return decorator


class GraphModuleCalibrationWrapper(EagerEvalWrapper):
    """
    A wrapper class for calibration
    """

    def __init__(  # noqa: C901
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
        seq_mse_candidates: int,
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
        self.seq_mse_candidates = seq_mse_candidates

    def _model_call(self, inps):
        all_logits = None
        kwargs = {}
        if self._use_kv_cache:
            kwargs["ar_len"] = self.ar_len
            kwargs["kv_updater"] = self.kv_updater
            kwargs["seq_mse_candidates"] = self.seq_mse_candidates

        all_logits = INFERENCE_REGISTRY[self._use_kv_cache](
            self.get_example_inputs,
            inps,
            self._model,
            self._tokenizer,
            max_seq_len=self.max_seq_length,
            use_i64_token=self.use_i64_token,
            collect_logits=True,
            **kwargs,
        )
        # one shot is enough for seq mse
        self.seq_mse_candidates = 0
        return all_logits


class LookaheadDecoder:
    """
    Lookahead decoding to speed up calibration
    """

    class NgramPool:
        def __init__(self, num_verifications: int):
            self.pool = defaultdict(OrderedDict)
            # keep the amount of ngrams as number of verification branches for simplicity
            self.num_verifications = num_verifications

        def add(self, ngram: Tuple[int]):
            key = ngram[0]
            # since there is no OrderedSet in python, use OrderedDict with dummy value 1
            self.pool[key][ngram[1:]] = 1
            if len(self.pool[key]) > self.num_verifications:
                # remove cache in FIFO fashion
                self.pool[key].popitem(last=False)

        def __getitem__(self, key):
            return self.pool[key]

        def __iter__(self):
            return iter(self.pool)

    def __init__(
        self,
        window_size: int,
        ngram_size: int,
        num_verifications: int,
        ar_size: int,
        mask_value: int,
    ):
        if ar_size < (ngram_size - 1) * (window_size + num_verifications):
            raise ValueError(
                "AR length is not enough to meet requirement. "
                "Should be at least (ngram_size - 1) * (window_size + num_verifications)."
            )

        self.window_size = window_size
        self.ngram_size = ngram_size
        self.ngram_pool = self.NgramPool(num_verifications)
        self.num_verifications = num_verifications
        self.verification_offset = window_size * (ngram_size - 1)
        self.ar_size = ar_size
        self.mask_value = mask_value

    @property
    def attention_mask(self) -> torch.Tensor:
        mask = torch.full((self.ar_size,) * 2, self.mask_value)
        lookahead_branch_mask = torch.triu(
            torch.full((self.window_size,) * 2, self.mask_value),
            diagonal=1,
        )
        for i in range(self.ngram_size - 1):
            mask[
                i * self.window_size : (i + 1) * self.window_size,
                : self.window_size,
            ] = lookahead_branch_mask
            for j in range(1, i + 1):
                mask[
                    i * self.window_size : (i + 1) * self.window_size,
                    j * self.window_size : (j + 1) * self.window_size,
                ].fill_diagonal_(0)

        verification_branch_mask = torch.triu(
            torch.full((self.ngram_size - 1,) * 2, self.mask_value),
            diagonal=1,
        )
        for i in range(self.num_verifications):
            indices = [i * (self.ngram_size - 1), (i + 1) * (self.ngram_size - 1)]
            slices = (slice(*[ind + self.verification_offset for ind in indices]),) * 2
            mask[slices] = verification_branch_mask
        mask[
            : self.verification_offset + (self.ngram_size - 1) * self.num_verifications,
            0,
        ] = 0

        return mask

    @property
    def position_offset(self) -> torch.Tensor:
        offsets = torch.zeros(self.ar_size, dtype=torch.int32)
        idx = 0
        # lookahead branches
        for i in range(self.ngram_size - 1):
            for j in range(self.window_size):
                offsets[idx] = i + j
                idx += 1

        # verification branches
        for _ in range(self.num_verifications):
            for j in range(1, self.ngram_size):
                offsets[idx] = j
                idx += 1

        return offsets

    def update_verification_branch(self, guess_token: int, inputs: List[int]) -> None:
        for branch, ngram in enumerate(self.ngram_pool[guess_token]):
            verification_offset = self.verification_offset + branch * (
                self.ngram_size - 1
            )
            for i, token in enumerate(ngram):
                inputs[verification_offset + i] = token

    def update_lookahead_branch(self, inputs: List[int], outputs: List[int]) -> None:
        # 1 level shifting
        for i in range(self.ngram_size - 2):
            for j in range(self.window_size):
                inputs[self.window_size * i + j] = inputs[
                    self.window_size * (i + 1) + j
                ]

        last_ngram_offset = self.window_size * (self.ngram_size - 2)
        for i in range(self.window_size):
            inputs[last_ngram_offset + i] = outputs[last_ngram_offset + i]

    def update_ngram_pool(self, inputs: List[int], outputs: List[int]) -> None:
        for i in range(self.window_size):
            ngram = [inputs[i]]
            for j in range(1, self.ngram_size - 1):
                ngram.append(inputs[i + j * self.window_size])

            ngram.append(outputs[i + self.window_size * (self.ngram_size - 2)])
            self.ngram_pool.add(tuple(ngram))

    def verify(
        self, inputs: List[int], outputs: List[int]
    ) -> Tuple[List[int], Optional[int]]:
        best_match, branch = [], None
        for i in range(self.num_verifications):
            current_match = [outputs[0]]
            verification_branch_offset = (
                self.verification_offset + (self.ngram_size - 1) * i
            )
            for j in range(self.ngram_size - 1):
                if inputs[verification_branch_offset + j] == current_match[-1]:
                    current_match.append(outputs[verification_branch_offset + j])
                else:
                    break

            if len(current_match[1:]) > len(best_match):
                best_match = current_match[1:]
                branch = i

        return best_match, branch


class QnnRunnerEvalWrapper(EagerEvalWrapper):
    """
    A wrapper class to run PPL scores with QNN on device.
    """

    def __init__(  # noqa: C901
        self,
        args,
        pte_path: str,
        tokenizer: Union[
            SentencePieceTokenizer, TiktokenTokenizer, HuggingFaceTokenizer
        ],
        runtime_tokenizer_path,
    ):
        self.args = args
        self.pte_path = pte_path
        self.enable_x86_64 = args.enable_x86_64
        self.max_seq_length = args.max_seq_len

        if self.enable_x86_64:
            logging.warning(
                "Using x86_64 emulator is NOT recommended as it is for CI purpose."
            )

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
        if pte_max_seq_len != self.max_seq_length:
            logging.warning(
                f"The pte provided has a max_seq_len {pte_max_seq_len}, which is different from --max_seq_len {self.max_seq_length} provided to the script, please ensure this is desired."
            )
            if pte_max_seq_len < self.max_seq_length:
                logging.warning(
                    f"The pte max_seq_len {pte_max_seq_len} is used since it is shorter than --max_seq_len {self.max_seq_length}"
                )
                self.max_seq_length = pte_max_seq_len
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
            target=args.target,
        )

        # collect output data
        output_data_folder = f"{self.args.artifact}/outputs"
        make_output_dir(output_data_folder)

        if not self.enable_x86_64:
            self.adb.push(inputs=[], files=[self.runtime_tokenizer_path])
        # n seq len = n-1 cache len, so we len(inps) = n-1 during _model_call
        # pyre-ignore
        super().__init__(None, tokenizer, self.max_seq_length - 1)

    def _model_call(self, inps):

        input_file_name = f"{self.args.artifact}/input_tokens.raw"
        inps = inps.to(torch.uint64).numpy()
        inps.tofile(input_file_name)

        outputs_path = "outputs/outputs.txt"
        dump_logits_path = "outputs/all_logit.raw"
        performance_output_path = "outputs/inference_speed.txt"
        output_tensor_list = []

        def post_process():
            with open(f"{self.args.artifact}/{dump_logits_path}", "r") as f:
                logits_dtype = np.float32 if self.kv_io_bit_width == 32 else np.uint16
                output_tensor = torch.from_numpy(
                    np.fromfile(f.name, dtype=logits_dtype).reshape(
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

        if self.enable_x86_64:
            qnn_sdk = os.getenv("QNN_SDK_ROOT")
            target = "x86_64-linux-clang"
            runner_cmd = " ".join(
                [
                    f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{self.args.build_folder}/lib &&",
                    f"./{self.args.build_folder}/examples/qualcomm/oss_scripts/llama/qnn_llama_runner",
                    f"--decoder_model_version {DECODER_MODEL_VERSION[self.args.decoder_model]}",
                    f"--tokenizer_path {self.runtime_tokenizer_path}",
                    f"--model_path {self.pte_path}",
                    f"--seq_len {self.max_seq_length}",
                    f"--output_path {self.args.artifact}/outputs/outputs.txt",
                    f"--performance_output_path {self.args.artifact}/{performance_output_path}",
                    f"--eval_mode {EVAL_MODE[self.args.model_mode]}",
                    "--temperature 0",
                    "--kv_updater ShiftPointer",
                    f"--dump_logits_path {self.args.artifact}/{dump_logits_path}",
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
            post_process()

        else:
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
            self.adb.pull(output_path=self.output_dir, callback=post_process)
        return output_tensor_list[0]


def smart_mask_updater(
    n_updates: int,
    atten_mask: AttentionMask,
    pos,
    k_caches,
    v_caches,
    new_k_caches,
    new_v_caches,
    # lookahead decoding related
    lade_token_offset=None,
    lade_pos_offset=None,
):
    # ar_len is unused in smart mask
    max_cache_len = k_caches[0].size(-1)

    if pos + n_updates <= max_cache_len:
        if lade_token_offset is not None:
            # lookahead decode update
            for i, offset in enumerate(lade_token_offset):
                current_pos = pos + i
                for j, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
                    k_cache[:, :, current_pos] = new_k_caches[j][:, :, offset]
                    v_cache[:, current_pos, :] = new_v_caches[j][:, offset, :]
        else:
            for i, k_cache in enumerate(k_caches):
                k_cache[:, :, pos : pos + n_updates] = new_k_caches[i][:, :, :n_updates]
            for i, v_cache in enumerate(v_caches):
                v_cache[:, pos : pos + n_updates, :] = new_v_caches[i][:, :n_updates, :]

        atten_mask.smart_mask_update(pos, n_updates, lade_pos_offset)

    pos += n_updates
    return pos, k_caches, v_caches


def shift_pointer_updater(
    n_updates: int,
    atten_mask: AttentionMask,
    pos,
    k_caches,
    v_caches,
    new_k_caches,
    new_v_caches,
    # lookahead decoding related
    lade_token_offset=None,
    lade_pos_offset=None,
):
    max_cache_len = k_caches[0].size(-1)
    if pos + n_updates <= max_cache_len:
        if lade_token_offset is not None:
            # lookahead decode update
            for offset in lade_token_offset:
                for i, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
                    k_caches[i] = torch.cat(
                        [
                            k_cache[:, :, 1:],
                            new_k_caches[i][:, :, offset].unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    v_caches[i] = torch.cat(
                        [v_cache[:, 1:, :], new_v_caches[i][:, offset, :].unsqueeze(1)],
                        dim=1,
                    )
        else:
            k_caches = [
                torch.cat(
                    [k_cache[:, :, n_updates:], new_k_caches[i][:, :, :n_updates]],
                    dim=-1,
                )
                for i, k_cache in enumerate(k_caches)
            ]
            v_caches = [
                torch.cat(
                    [v_cache[:, n_updates:, :], new_v_caches[i][:, :n_updates, :]],
                    dim=1,
                )
                for i, v_cache in enumerate(v_caches)
            ]

        atten_mask.shift_pointer_update(pos, n_updates, lade_pos_offset)

    pos += n_updates
    return pos, k_caches, v_caches


@register_inference(use_kv_cache=True)
def kv_inference(  # noqa: C901
    get_example_inputs,
    prompt: Union[str, list],
    module: torch.fx.GraphModule,
    tokenizer,
    ar_len=1,
    max_seq_len=512,
    kv_updater=smart_mask_updater,
    use_i64_token=False,
    collect_logits=False,
    seq_mse_candidates=0,
    lookahead_config=None,
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
                *atten_mask,
                tmp_pos,
                *k_caches,
                *v_caches,
            )
            if collect_logits:
                result_logits.append(logits[:, :num_tokens_in_chunk])

            # We should have enough calibration data when generating last token if task was specified
            if seq_mse_candidates != 0 and pos == num_prompt_tokens - 1:
                with SeqMSE(module, seq_mse_candidates):
                    module(
                        tmp_token_list,
                        *atten_mask,
                        tmp_pos,
                        *k_caches,
                        *v_caches,
                    )

            # Update the pos, KV cache and attention mask.
            pos, k_caches, v_caches = kv_updater(
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
        if lookahead_config is None:
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
                tmp_pos[0, :num_tokens_in_chunk] = all_pos[
                    0, chunk_start_idx:chunk_end_idx
                ]

                logits, new_k_caches, new_v_caches = module(
                    tmp_token_list,
                    *atten_mask,
                    tmp_pos,
                    *k_caches,
                    *v_caches,
                )

                pos, k_caches, v_caches = kv_updater(
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
        else:
            # TODO: support batch decode if necessary
            # variable declaration
            window, ngram, gcap = lookahead_config
            lade = LookaheadDecoder(
                window_size=window,
                ngram_size=ngram,
                num_verifications=gcap,
                ar_size=ar_len,
                mask_value=next(iter(atten_mask)).min().item(),
            )
            generated_tokens, accepted_tokens = 0, 0
            input_tokens = [total_token_list[-1]] * ar_len
            pos_offsets = lade.position_offset.unsqueeze(0)
            pos_offsets_list = pos_offsets.flatten().tolist()
            # replace ar attention mask to lookahead version
            for mask in atten_mask:
                mask[:, :, -ar_len:] = lade.attention_mask.unsqueeze(0)
            # start decoding
            while (
                total_token_list[-1] != tokenizer.eos_id
                and len(total_token_list) < max_cache_len
            ):
                # populate verification branch
                lade.update_verification_branch(
                    guess_token=input_tokens[0],
                    inputs=input_tokens,
                )
                # inference
                logits, new_k_caches, new_v_caches = module(
                    torch.tensor(input_tokens, dtype=dtype).unsqueeze(0),
                    *atten_mask,
                    pos_offsets + pos,
                    *k_caches,
                    *v_caches,
                )
                # collect outputs
                output_tokens = torch.argmax(logits, dim=-1).flatten().tolist()
                # update ngram pool
                lade.update_ngram_pool(inputs=input_tokens, outputs=output_tokens)
                # try matching verification branches
                best_match, branch_no = lade.verify(
                    inputs=input_tokens, outputs=output_tokens
                )
                # check if any match was found
                lade_token_offset, num_match = [0], len(best_match)
                if num_match > 0:
                    accepted_tokens += num_match
                    lade_token_offset += [
                        e + lade.verification_offset + branch_no * (ngram - 1)
                        for e in range(num_match)
                    ]
                # update kv cache
                pos, k_caches, v_caches = kv_updater(
                    len(lade_token_offset),
                    atten_mask,
                    pos,
                    k_caches,
                    v_caches,
                    new_k_caches,
                    new_v_caches,
                    lade_token_offset,
                    pos_offsets_list,
                )
                generated_tokens += len(lade_token_offset)
                # update lookahead branch
                lade.update_lookahead_branch(inputs=input_tokens, outputs=output_tokens)
                # update token list
                for token in [output_tokens[0], *best_match]:
                    total_token_list.append(token)
                    if token == tokenizer.eos_id:
                        break
                # fill next input token
                input_tokens[0] = total_token_list[-1]

            logging.info(
                f"lookahead accepted / total generated: {accepted_tokens} / {generated_tokens}"
            )

    logging.info(f"kv inference result:\n{tokenizer.decode(total_token_list)}")
    if collect_logits:
        result_logits = torch.cat(result_logits, dim=1)
    return result_logits


@register_inference(use_kv_cache=False)
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
            results = module(tmp_token_list, *atten_mask)
            if len(results) == 3:
                logits, _, _ = results
            elif len(results) == 1:
                logits = results
            token = torch.argmax(logits[:, pos - 1], dim=-1).item()
            token_list.append(token)
            if collect_logits:
                result_logits = logits[:, :pos]
            pos += 1
    if isinstance(prompt, str):
        logging.info(f"prefill inference result:\n{tokenizer.decode(token_list)}")
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
    seq_mse_candidates: int = 0,
    lookahead_config: Optional[Tuple[int]] = None,
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
        kwargs = {}
        if use_kv_cache:
            kwargs["ar_len"] = ar_len
            kwargs["kv_updater"] = kv_updater
            kwargs["lookahead_config"] = lookahead_config

        INFERENCE_REGISTRY[use_kv_cache](
            get_example_inputs,
            prompt,
            module,
            tokenizer,
            max_seq_len=max_seq_len,
            use_i64_token=use_i64_token,
            collect_logits=False,
            **kwargs,
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
            seq_mse_candidates=seq_mse_candidates,
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
