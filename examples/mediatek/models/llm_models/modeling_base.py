from abc import ABC, abstractmethod

import torch


class BaseModelChunk(ABC, torch.nn.Module):
    def __init__(
        self,
        config,
        num_blocks,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        return_attn=False,
        jit_trace=False,
    ):
        torch.nn.Module.__init__(self)
        torch.set_default_dtype(dtype)
        self.dtype = dtype
        self.config = config
        self.num_blocks = num_blocks
        self.chunk_idx = chunk_idx
        self.include_tail = include_tail
        self.return_attn = return_attn
        self.jit_trace = jit_trace
        self.device_list = []

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def load_weights(self, state_dict, state_dict_start_idx, verbose):
        pass

    @abstractmethod
    def get_example_inputs(self, num_token, cache_size, get_dym_shape):
        pass
