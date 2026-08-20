# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import torch


class TestConfig:
    host_ip: str = "111.111.111.111"
    chipset: str = "E9955"


class GreedyLM:
    def __init__(self, vocab, blank_label="*"):
        self.vocab = vocab
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.blank_label = blank_label

    def encode(self, text):
        return [self.char_to_id[c] for c in text.lower()]

    def decode_ids(self, ids):
        if ids.ndim == 2:  # batch|steps
            return [self.decode_ids(t) for t in ids]

        decoded_text = "".join([self.vocab[id] for id in ids])

        return decoded_text

    def decode_ctc(self, emissions):
        if emissions.ndim == 3:  # batch|labels|steps
            return [self.decode_ctc(t) for t in emissions]

        amax_ids = emissions.argmax(0)
        amax_ids_collapsed = torch.unique_consecutive(amax_ids)
        decoded_text = "".join([self.vocab[id] for id in amax_ids_collapsed])
        decoded_text = decoded_text.replace(self.blank_label, "")

        return decoded_text
