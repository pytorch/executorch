# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT

import unittest

import pkg_resources

from pytorch_tokenizers.tiktoken import TiktokenTokenizer


class TestTiktokenTokenizer(unittest.TestCase):
    def test_default(self):
        model_path = pkg_resources.resource_filename(
            "pytorch.tokenizers.test", "test_tiktoken_tokenizer.model"
        )
        tiktoken = TiktokenTokenizer(model_path)
        s = "<|begin_of_text|> hellow world."
        self.assertEqual(s, tiktoken.decode(tiktoken.encode(s, bos=False, eos=False)))

    def test_custom_pattern_and_special_tokens(self):
        o220k_pattern = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        model_path = pkg_resources.resource_filename(
            "pytorch.tokenizers.test", "test_tiktoken_tokenizer.model"
        )
        tiktoken = TiktokenTokenizer(
            model_path,
            pat_str=o220k_pattern,
            special_tokens=[
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|custom_token|>",
            ],
        )
        custom_token_id = tiktoken.special_tokens["<|custom_token|>"]

        s = "<|begin_of_text|> hellow world, this is a custom token: <|custom_token|>."
        encoding = tiktoken.encode(
            s,
            bos=False,
            eos=False,
            allowed_special="all",
        )
        self.assertTrue(custom_token_id in encoding)
        self.assertEqual(s, tiktoken.decode(encoding))
