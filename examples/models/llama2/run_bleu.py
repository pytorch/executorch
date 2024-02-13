# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import platform
from dataclasses import dataclass
from typing import Tuple

import sacrebleu
import torch

from fairseq2.data import VocabularyInfo
from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
)

from fairseq2.generation import BeamSearchSequenceGenerator, NGramRepeatBlockProcessor
from fairseq2.models.llama.builder import (
    create_llama_model,
    LLaMAConfig,
    TransformerDecoderModel,
)


logging_format = f"%(asctime)s - {platform.node()} - %(process)s - %(levelname)s - %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
)

logger = logging.getLogger("langllama")


EXAMPLE = "Dado que el cielo se mantendrá oscuro casi durante todo el día, es una gran oportunidad para contemplar las auroras boreales."
SUPPORTED_LANGS = ["ru", "en", "pt", "id", "fr", "de", "es", "zh", "hi", "ar", "ko"]
LANG_MAPPING = {
    "rus": "ru",
    "eng": "en",
    "spa": "es",
}

SPM_PATH = os.path.expanduser("~/translate_llama/flores200sacrebleuspm")
CHECKPOINT_PATH = os.path.expanduser("~/translate_llama/checkpoint_last.pt")
TEST_DATASET = os.path.expanduser("~/translate_llama/flores_devtest_spa-eng.tsv")
CUDA_DEVICE = torch.device("cuda:0")
DTYPE = torch.float32


@dataclass
class Sample:
    src_lang: str
    tgt_lang: str
    src_text: str
    tgt_text: str


def load_test_samples():
    with open(TEST_DATASET, "r") as fp_in:
        headers = fp_in.readline()[:-1].split("\t")
        src_lang_idx, tgt_lang_idx = headers.index("src_lang"), headers.index(
            "tgt_lang"
        )
        src_text_idx, tgt_text_idx = headers.index("src_text"), headers.index(
            "tgt_text"
        )
        for line in fp_in:
            cols = line[:-1].split("\t")
            yield Sample(
                src_lang=cols[src_lang_idx],
                tgt_lang=cols[tgt_lang_idx],
                src_text=cols[src_text_idx],
                tgt_text=cols[tgt_text_idx],
            )


def load_tokenizer(path: str = SPM_PATH) -> Tuple[SentencePieceModel, VocabularyInfo]:
    model = SentencePieceModel(path, ["<pad>@0"])
    vocab_info = VocabularyInfo(
        256206,  # model.vocabulary_size,
        model.unk_idx,
        model.bos_idx,
        model.eos_idx,
        model.pad_idx,
    )
    return model, vocab_info


def load_model_(
    vocab_info: VocabularyInfo,
    checkpoint_path: str = CHECKPOINT_PATH,
    device: torch.device = CUDA_DEVICE,
    dtype: torch.dtype = DTYPE,
):
    embed_dim = 2048
    model_config = LLaMAConfig(
        model_dim=embed_dim,
        max_seq_len=4096,
        vocab_info=vocab_info,
        num_layers=8,
        num_attn_heads=16,
        num_key_value_heads=16,
        ffn_inner_dim=embed_dim * 4,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )
    model = create_llama_model(config=model_config, dtype=dtype, device=device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def translate(
    model: TransformerDecoderModel,
    spm_model: SentencePieceModel,
    vocab_info: VocabularyInfo,
    src_text: str,
    target_lang: str,
    device: torch.device = CUDA_DEVICE,
):
    eos_idx = vocab_info.eos_idx
    encoded_src_text = torch.cat(
        [
            torch.LongTensor([eos_idx]),
            SentencePieceEncoder(model=spm_model)(target_lang + " " + src_text),
            torch.LongTensor([eos_idx, eos_idx]),
        ],
        dim=0,
    ).to(device)
    encoded_src_text = encoded_src_text[
        encoded_src_text != vocab_info.unk_idx
    ]  # strip unks (optional)
    src_len = encoded_src_text.shape[0]
    max_gen_len = int(src_len * 1.1)
    generator = BeamSearchSequenceGenerator(
        model=model,
        beam_size=5,
        echo_prompt=True,
        max_gen_len=max_gen_len,
        max_seq_len=src_len + max_gen_len,
        len_penalty=0.5,
        step_processors=[NGramRepeatBlockProcessor(ngram_size=4)],
    )
    with torch.inference_mode():
        output = generator(
            prompt_seqs=encoded_src_text.unsqueeze(0), prompt_padding_mask=None
        )
    ouput_seq = output.hypotheses[0][0].seq[src_len:]
    assert ouput_seq.shape[-1] <= max_gen_len + 1
    translation = str(SentencePieceDecoder(spm_model)(ouput_seq))
    return translation


def eval_bleu(model, spm_model, vocab_info, max_test_samples: int = 100) -> float:
    logger.info(f"Model: {model}")
    ref_translations = []
    translations = []
    for idx, sample in enumerate(list(load_test_samples())[:max_test_samples]):
        translated = translate(
            model=model,
            spm_model=spm_model,
            vocab_info=vocab_info,
            src_text=sample.src_text,
            target_lang=LANG_MAPPING[sample.tgt_lang],
        )
        translations.append(translated)
        ref_translations.append(sample.tgt_text)
        logger.info(f"[{str(idx).zfill(3)}] SRC: {sample.src_text}")
        logger.info(f"[{str(idx).zfill(3)}] TGT: {translated}")
        logger.info(f"[{str(idx).zfill(3)}] REF: {sample.tgt_text}")
    chrf_bleu = sacrebleu.corpus_chrf(translations, [ref_translations]).score
    logger.info(f"ChrF++ BLEU: {chrf_bleu}")
    return chrf_bleu


def eval_size(
    parameters,
) -> float:
    if parameters["dtype"] == 0:
        fp_size = 4
    else:
        fp_size = 2

    tok_embeddings_els = 524709888
    layer_linear_els = 411060384
    layer_norm_els = 16224
    output_linear_els = 524709888

    embed_size = fp_size
    tok_embed_scale_size = 0
    if parameters["embedding_type"] > 0:
        embed_size = 1 / parameters["embedding_type"]
        group_size = 2 ** parameters["embedding_groupsize"]
        tok_embed_scale_size = (tok_embeddings_els // group_size) * fp_size
    embed_weight_size = tok_embeddings_els * embed_size

    linear_size = fp_size
    linear_scale_size = 0
    if parameters["linear_type"] > 0:
        linear_size = 1 / parameters["linear_type"]
        group_size = 2 ** parameters["linear_groupsize"]
        linear_scale_size = (layer_linear_els // group_size) * fp_size
    linear_weight_size = layer_linear_els * linear_size

    layer_norm_weight_size = layer_norm_els * fp_size

    output_linear_size = fp_size
    output_linear_scale_size = 0
    if parameters["output_linear_type"] > 0:
        output_linear_size = 1 / parameters["output_linear_type"]
        group_size = 2 ** parameters["output_linear_groupsize"]
        output_linear_scale_size = (output_linear_els // group_size) * fp_size
    output_linear_weight_size = layer_linear_els * output_linear_size

    total_size = (
        embed_weight_size
        + tok_embed_scale_size
        + linear_weight_size
        + linear_scale_size
        + layer_norm_weight_size
        + output_linear_weight_size
        + output_linear_scale_size
    )

    return total_size
