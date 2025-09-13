# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

from models.llm_models.configuration_base import BaseConfig


# flake8: noqa: C901


class WhisperEncoderConfig(BaseConfig):
    def __init__(
        self,
        model_type="whisper",
        encoder_ffn_dim=None,
        encoder_layers=None,
        encoder_attention_heads=None,
        d_model=None,
        decoder_layers=None,
        num_mel_bins=None,
        max_source_positions=None,
        max_position_embeddings=None,
        head_dim=None,
        response_handler=None,
        **kwargs,
    ):
        super().__init__()
        self.model_type = model_type
        self.intermediate_size = encoder_ffn_dim
        if self.intermediate_size is None:
            raise KeyError("encoder_ffn_dim is required but missing from config.json")

        self.num_hidden_layers = encoder_layers
        if self.num_hidden_layers is None:
            raise KeyError("encoder_layers is required but missing from config.json")

        self.num_attention_heads = encoder_attention_heads
        if self.num_attention_heads is None:
            raise KeyError(
                "encoder_attention_heads is required but missing from config.json"
            )

        self.hidden_size = d_model
        if self.hidden_size is None:
            raise KeyError("d_model is required but missing from config.json")

        self.decoder_num_layers = decoder_layers
        if self.decoder_num_layers is None:
            raise KeyError("decoder_layers is required but missing from config.json")

        self.num_mel_bins = num_mel_bins
        if self.num_mel_bins is None:
            raise KeyError("num_mel_bins is required but missing from config.json")

        self.max_source_positions = max_source_positions
        if self.max_source_positions is None:
            raise KeyError(
                "max_source_positions is required but missing from config.json"
            )

        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        else:
            self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.norm = kwargs.pop("norm", "LayerNorm")
        self.num_key_value_heads = kwargs.pop(
            "num_key_value_heads", self.num_attention_heads
        )

        if response_handler is None:
            response_handler = nullcontext()
        if kwargs.pop("verbose", True):
            self.print_config(response_handler)

    def print_config(self, response_handler):
        with response_handler:
            print(f"{self.model_type} encoder config:")
            print(f"Hidden size:          {self.hidden_size}")
            print(f"Intermediate size:    {self.intermediate_size}")
            print(f"Num layers:           {self.num_hidden_layers}")
            print(f"Num attention heads:  {self.num_attention_heads}")
            print(f"Head dim:             {self.head_dim}")
            print(f"Decoder num layers:   {self.decoder_num_layers}")
            print(f"Max source pos:       {self.max_source_positions}")
            if self.max_position_embeddings is not None:
                print(f"Max pos emb:          {self.max_position_embeddings}")
            print(f"Num mel bins:         {self.num_mel_bins}")
            print()


class WhisperDecoderConfig(BaseConfig):
    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        intermediate_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        max_position_embeddings=None,
        norm="LayerNorm",
        position_embedding="rope",
        norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=0,
        use_stable_embedding=False,
        tie_word_embeddings=True,
        combine_qkv=False,
        response_handler=None,
        model_type=None,
        **kwargs,
    ):
        super().__init__()

        self.model_type = model_type
        self.vocab_size = vocab_size
        if self.vocab_size is None:
            raise KeyError("vocab_size is required but missing from config.json")
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            raise KeyError("hidden_size is required but missing from config.json")
        self.intermediate_size = intermediate_size
        if self.intermediate_size is None:
            raise KeyError("intermediate_size is required but missing from config.json")
        self.num_hidden_layers = num_hidden_layers
        if self.num_hidden_layers is None:
            raise KeyError("num_hidden_layers is required but missing from config.json")
        self.num_attention_heads = num_attention_heads
        if self.num_attention_heads is None:
            raise KeyError(
                "num_attention_heads is required but missing from config.json"
            )
        self.num_key_value_heads = kwargs.pop(
            "num_key_value_heads", self.num_attention_heads
        )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise RuntimeError(
                f"num_attention_heads ({self.num_attention_heads}) must be exactly "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        if norm not in ["RMSNorm", "LayerNorm"]:
            raise ValueError("norm must be one of: RMSNorm (default) or LayerNorm")
        self.norm = norm
        self.norm_eps = kwargs.pop("rms_norm_eps", norm_eps)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.head_dim = kwargs.pop(
            "head_dim", self.hidden_size // self.num_attention_heads
        )

        if position_embedding not in ["rope", "alibi"]:
            raise ValueError("Positional embedding must be one of: rope, alibi")
        self.position_embedding = position_embedding
        self.ntk_scaling_factor = kwargs.pop("ntk_scaling_factor", 1.0)
        if self.ntk_scaling_factor != 1.0 and self.position_embedding != "rope":
            raise KeyError("ntk_scaling_factor is strictly for position_embedding=rope")
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings is None and self.position_embedding == "rope":
            raise KeyError(
                "max_position_embeddings is required for position_embedding=rope but missing from config.json"
            )

        self.use_stable_embedding = use_stable_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.combine_qkv = combine_qkv

        self.tokenizer = kwargs.pop("tokenizer", self.tokenizer)
        self.max_source_positions = kwargs.pop("max_source_positions", None)
        if self.max_source_positions is None:
            raise KeyError(
                "max_source_positions is required but missing from config.json"
            )

        if response_handler is None:
            response_handler = nullcontext()
        if kwargs.pop("verbose", True):
            self.print_config(response_handler)

    def print_config(self, response_handler):
        with response_handler:
            print(f"{self.model_type} llm config:")
            print(f"Hidden size:          {self.hidden_size}")
            print(f"Intermediate size:    {self.intermediate_size}")
            print(f"Num layers:           {self.num_hidden_layers}")
            print(f"Num attention heads:  {self.num_attention_heads}")
            print(f"Num KV heads:         {self.num_key_value_heads}")
            print(f"Head Dim:             {self.head_dim}")
            print(f"Positional embedding: {self.position_embedding}")
            if self.position_embedding == "rope":
                print(f"Max pos emb:          {self.max_position_embeddings}")
                if self.ntk_scaling_factor != 1.0:
                    print(f"NTK scaling factor:   {self.ntk_scaling_factor}")
            print(f"Norm type:            {self.norm}")
            print(f"Norm epsilon:         {self.norm_eps}")
            print(f"BOS token id:         {self.bos_token_id}")
            print(f"EOS token id:         {self.eos_token_id}")
            print(f"PAD token id:         {self.pad_token_id}")
            print(f"UNK token id:         {self.unk_token_id}")
            print(f"Vocab size:           {self.vocab_size}")
            print(f"Use stable embedding: {self.use_stable_embedding}")
            print(f"Use qk norm:          {self.use_qk_norm}")
            print(f"Tie word embeddings:  {self.tie_word_embeddings}")
            print(f"Combine QKV:          {self.combine_qkv}")
            if self.tokenizer != "default":
                print(f"Tokenizer:            {self.tokenizer}")
            print()


class WhisperPreprocessorConfig:
    def __init__(self, response_handler=None, **kwargs):
        self.model_type = kwargs.pop("model_type", "whisper")
        assert self.model_type == (
            "whisper"
        ), f"Expected model_type to be whisper but got {self.model_type} instead"

        self.feature_size = kwargs.pop("feature_size", 80)
        self.sampling_rate = kwargs.pop("sampling_rate", 16000)
        self.return_attention_mask = kwargs.pop("return_attention_mask", False)

        if response_handler is None:
            response_handler = nullcontext()
        if kwargs.pop("verbose", True):
            self.print_config(response_handler)

    def print_config(self, response_handler):
        with response_handler:
            print(f"{self.model_type} preprocessor config:")
            print(f"Feature Size:       {self.feature_size}")
            print(f"Sampling Rate:      {self.sampling_rate}")
            print(f"Audio Attn Mask:    {self.return_attention_mask}")
            print()


class WhisperConfig(BaseConfig):
    def __init__(self, **kwargs):
        encoder_config = kwargs.pop("encoder", None)
        if encoder_config is None:
            raise KeyError("Encoder config not found in config.json")
        llm_config = kwargs.pop("llm", None)
        if llm_config is None:
            raise KeyError("llm config not found in config.json")
        p_config = kwargs.pop("preprocessor", None)
        if p_config is None:
            raise KeyError("preprocessor config not found in config.json")

        self.encoder = WhisperEncoderConfig(**encoder_config, verbose=False)
        self.llm = WhisperDecoderConfig(**llm_config, verbose=False)
        self.p = WhisperPreprocessorConfig(**p_config, verbose=False)
        self.model_type = "whisper"
        self.tokenizer = "default"

        response_handler = kwargs.pop("response_handler", None)
        if response_handler is None:
            response_handler = nullcontext()

        if kwargs.pop("verbose", True):
            self.print_config(response_handler)

    def print_config(self, response_handler):
        self.encoder.print_config(response_handler)
        self.llm.print_config(response_handler)
        self.p.print_config(response_handler)
