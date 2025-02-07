from abc import ABC, abstractmethod


class BaseConfig(ABC):
    def __init__(self):
        self.model_type = "base"
        self.vocab_size = None
        self.hidden_size = 0
        self.intermediate_size = 0
        self.num_hidden_layers = 0
        self.num_attention_heads = 0
        self.num_key_value_heads = 0
        self.position_embedding = None
        self.max_position_embeddings = 0
        self.ntk_scaling_factor = 1.0
        self.norm = None
        self.norm_eps = 0

        self.bos_token_id = None
        self.pad_token_id = None
        self.eos_token_id = None
        self.unk_token_id = None

        self.use_stable_embedding = False
        self.tie_word_embeddings = False
        self.combine_qkv = False

        self.tokenizer = "default"

    @abstractmethod
    def print_config(self, response_handler):
        pass
