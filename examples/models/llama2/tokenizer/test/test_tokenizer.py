# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import struct
import tempfile
import unittest
from unittest.mock import patch

from executorch.examples.models.llama2.tokenizer.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    @patch(
        "executorch.examples.models.llama2.tokenizer.tokenizer.SentencePieceProcessor"
    )
    def test_export(self, mock_sp):
        # Set up the mock SentencePieceProcessor
        mock_sp.return_value.vocab_size.return_value = 0
        mock_sp.return_value.get_piece_size.return_value = 0
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=True) as temp:
            # Initialize the tokenizer with the temporary file as the model
            tokenizer = Tokenizer(temp.name)
            # Export the tokenizer to another temporary file
            with tempfile.NamedTemporaryFile(delete=True) as output:
                tokenizer.export(output.name)
                # Open the output file in binary mode and read the first 16 bytes
                with open(output.name, "rb") as f:
                    data = f.read(16)
                # Unpack the data as 4 integers
                vocab_size, max_token_length = struct.unpack("II", data)
                # Check that the integers match the properties of the tokenizer
                self.assertEqual(vocab_size, 0)
                # Check that the max token length is correct
                self.assertEqual(max_token_length, 0)
