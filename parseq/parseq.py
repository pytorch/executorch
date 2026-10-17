# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Sequence

import torch
from timm.models.helpers import named_apply

from timm.models.vision_transformer import PatchEmbed, VisionTransformer
from torch import nn, nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer
from torch.nn.utils.rnn import pad_sequence


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.gelu
        super().__setstate__(state)

    def forward_stream(
        self,
        tgt: Tensor,
        tgt_norm: Tensor,
        tgt_kv: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(
            tgt_norm,
            tgt_kv,
            tgt_kv,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm2(tgt))))
        )
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
        update_content: bool = True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(
            query,
            query_norm,
            content_norm,
            memory,
            query_mask,
            content_key_padding_mask,
        )[0]
        if update_content:
            content = self.forward_stream(
                content,
                content_norm,
                content_norm,
                memory,
                content_mask,
                content_key_padding_mask,
            )[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query,
                content,
                memory,
                query_mask,
                content_mask,
                content_key_padding_mask,
                update_content=not last,
            )
        query = self.norm(query)
        return query


class Encoder(VisionTransformer):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            num_classes=0,  # These
            global_pool="",  # disable the
            class_token=False,  # classifier head.
        )

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


def init_weights(module: nn.Module, name: str = "", exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class BaseTokenizer(ABC):

    def __init__(
        self, charset: str, specials_first: tuple = (), specials_last: tuple = ()
    ) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> list[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: list[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return "".join(tokens) if join else tokens

    @abstractmethod
    def encode(
        self, labels: list[str], device: Optional[torch.device] = None
    ) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, list[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(
        self, token_dists: Tensor, raw: bool = False
    ) -> tuple[list[str], list[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [
            self._stoi[s] for s in specials_first + specials_last
        ]

    def encode(
        self, labels: list[str], device: Optional[torch.device] = None
    ) -> Tensor:
        batch = [
            torch.as_tensor(
                [self.bos_id] + self._tok2ids(y) + [self.eos_id],
                dtype=torch.long,
                device=device,
            )
            for y in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, list[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[: eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids


class PARSeq(nn.Module):

    def __init__(
        self,
        tokenizer: Tokenizer,  # Include tokenizer so we can infer using this class directly
        max_label_length: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        decode_ar: bool,
        refine_iters: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        num_tokens = len(tokenizer)

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        self.encoder = Encoder(
            img_size,
            patch_size,
            embed_dim=embed_dim,
            depth=enc_depth,
            num_heads=enc_num_heads,
            mlp_ratio=enc_mlp_ratio,
        )
        decoder_layer = DecoderLayer(
            embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout
        )
        self.decoder = Decoder(
            decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim)
        )

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, num_tokens - 2)
        self.text_embed = TokenEmbedding(num_tokens, embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(
            torch.Tensor(1, max_label_length + 1, embed_dim)
        )
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=["encoder"]), self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

        self.export_mode = None
        self.register_buffer(
            "tmp",
            torch.triu(
                torch.ones(
                    self.max_label_length + 1,
                    self.max_label_length + 1,
                    dtype=torch.float,
                    device=self._device,
                ),
                2,
            ),
        )
        self.register_buffer(
            "tmp2",
            torch.triu(
                torch.ones(
                    self.max_label_length + 1,
                    self.max_label_length + 1,
                    dtype=torch.float,
                    device=self._device,
                ),
                1,
            ),
        )

    @property
    def _device(self) -> torch.device:
        return next(self.head.parameters(recurse=False)).device

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {"text_embed.embedding.weight", "pos_queries"}
        enc_param_names = {"encoder." + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(
            tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask
        )

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        tokenizer = self.tokenizer
        testing = max_length is None
        max_length = (
            self.max_label_length
            if max_length is None
            else min(max_length, self.max_label_length)
        )
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        if self.export_mode in (None, "executorch"):
            tgt_mask = query_mask = torch.triu(
                torch.ones(
                    (num_steps, num_steps), dtype=torch.bool, device=self._device
                ),
                1,
            )
        else:  # torch.bool unsupported by onnxruntime for "Where" operation. torch.float slightly reduces performance.
            # From PyTorch 1 version of PARSeq. Not compatible with executorch (does not like -inf).
            tgt_mask = query_mask = torch.triu(
                torch.full((num_steps, num_steps), float("-inf"), device=self._device),
                1,
            )

        if self.decode_ar:
            tgt_in = torch.full(
                (bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device
            )
            tgt_in[:, 0] = tokenizer.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)

                    # TODO: Efficient batch decoding check not compatible with export. Can we assume it to always be true?
                    if self.export_mode is not None:
                        break

                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full(
                (bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device
            )
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            if (
                self.export_mode == "dynamo"
            ):  # Requires concrete boolean indexing for JAX compatibility. # TODO: Performance loss?
                for x in range(2, num_steps):
                    for y in range(0, x - 1):
                        query_mask[y, x] = 0
            else:
                query_mask[
                    torch.triu(
                        torch.ones(
                            num_steps, num_steps, dtype=torch.bool, device=self._device
                        ),
                        2,
                    )
                ] = 0

            bos = torch.full(
                (bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device
            )
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # Mask tokens beyond the first EOS token.
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    pos_queries,
                    query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.head(tgt_out)

        if self.export_mode is not None:
            return logits.softmax(dim=2)

        return logits
