# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Script to rewrite tokenizer model given by sentencepiece, with lightweight
# postprocessing logic.

import argparse
import logging
import os
import struct
from typing import List

from sentencepiece import SentencePieceProcessor as SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(
            model_path
        ), f"Need a valid tokenizer model path but got {model_path}"
        # pyre-fixme[28]: Unexpected keyword argument `model_file` to call `SentencePieceProcessor.__init__`.
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        logging.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        # pyre-fixme[16]: `SentencePieceProcessor` has no attribute `get_piece_size`.
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        # pyre-fixme[16]: `SentencePieceProcessor` has no attribute `encode`.
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        # pyre-fixme[16]: `SentencePieceProcessor` has no attribute `encode`.
        return self.sp_model.decode(t)

    def export(self, output_path: str, *, prepend_padding: bool = False) -> None:
        """
        Export tokenizer.model to another serialization format. Here we did some lightweight
        processing such as supporting prepend padding token, prepend max token length and
        replace '_' back to empty space.

        The binary format is:
        1. vocab size: int32
        2. bos token id: int32
        3. eos token id: int32
        4. max token length: int32
        5. score: float32, len of bytes: int32, token bytes: [byte] for each token

        :param output_path: output path of the new binary.
        :param prepend_padding: a boolean to control if we want to prepend a padding token.

        :return: None
        """

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []

        if prepend_padding:
            # Here we use the default padding token and its score.
            tokens.append("<pad>".encode("utf-8"))
            scores.append(-1)

        for i in range(self.n_words):

            # decode the token and light postprocessing
            # pyre-fixme[16]: `SentencePieceProcessor` has no attribute `id_to_piece`.
            t = self.sp_model.id_to_piece(i)
            # pyre-fixme[16]: `SentencePieceProcessor` has no attribute `get_score`.
            s = self.sp_model.get_score(i)
            # sentencepiece use '<s>' as BOS and '</s>' for EOS
            if i == self.bos_id:
                t = "<s>"
            elif i == self.eos_id:
                t = "</s>"
            t = t.replace("▁", " ")  # sentencepiece uses this character as whitespace
            b = t.encode("utf-8")  # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = 0 if not tokens else max(len(t) for t in tokens)

        # write to a binary file
        with open(output_path, "wb") as f:
            # write the vocab size, bos/eos ids and max token length
            f.write(
                struct.pack(
                    "IIII", self.n_words, self.bos_id, self.eos_id, max_token_length
                )
            )
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)
        logging.info(f"Wrote tokenizer to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tokenizer-model",
        type=str,
        default="tokenizer.model",
        help="path to tokenizer model, given by sentencepiece",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=None,
        help="output path of postprocessed tokenizer model",
    )
    parser.add_argument(
        "-p",
        "--prepend-padding",
        action="store_true",
        help="whether to prepend a padding token to the beginning of the tokenizer",
    )

    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model)

    output_path = (
        args.output_path
        if args.output_path
        else args.tokenizer_model.replace(".model", ".bin")
    )
    t.export(output_path, prepend_padding=args.prepend_padding)
