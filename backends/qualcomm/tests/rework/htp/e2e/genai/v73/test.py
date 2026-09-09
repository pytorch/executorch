# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.backends.qualcomm.tests.rework.src.genai import *  # noqa: F403


def test_static_llm_model(request):
    StaticLLM.test(request)  # noqa: F405


def test_static_llm_eval_limit(request):
    StaticLLMEvalLimit.test(request)  # noqa: F405


def test_codegen2_1b(request):
    Codegen2.test(request)  # noqa: F405


def test_llama_stories_260k(request):
    LlamaStories260K.test(request)  # noqa: F405


def test_llama_stories_110m(request):
    LlamaStories110M.test(request)  # noqa: F405


def test_attention_sink(request):
    AttentionSink.test(request)  # noqa: F405


def test_hf_causal_lm(request):
    HFCausalLM.test(request)  # noqa: F405


def test_static_llm_qat(request):
    StaticLLMQAT.test(request)  # noqa: F405


def test_static_asr(request):
    StaticASR.test(request)  # noqa: F405


def test_static_vlm(request):
    StaticVLM.test(request)  # noqa: F405
