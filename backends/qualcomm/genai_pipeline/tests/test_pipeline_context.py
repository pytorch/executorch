# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from backends.qualcomm.genai_pipeline.pipeline_context import (
    PipelineContext,
    PipelineContextBuilder,
)

TEST_MODEL_NAME = "llama3_2-1b_instruct"
TEST_SOC_MODEL = "SM8750"
TEST_PROMPT = "What is AI?"
TEST_ARTIFACT_DIR = "/tmp/artifacts"
TEST_DEFAULT_ARTIFACT_DIR = "./genai_artifacts"


class TestPipelineContext(unittest.TestCase):

    def test_create_with_all_fields(self):
        ctx = PipelineContext(
            model_name=TEST_MODEL_NAME,
            soc_model=TEST_SOC_MODEL,
            prompt=["Hello world"],
            artifact_dir=TEST_ARTIFACT_DIR,
            extra_options={"key": "value"},
        )
        self.assertEqual(ctx.model_name, TEST_MODEL_NAME)
        self.assertEqual(ctx.soc_model, TEST_SOC_MODEL)
        self.assertEqual(ctx.prompt, ["Hello world"])
        self.assertEqual(ctx.artifact_dir, TEST_ARTIFACT_DIR)
        self.assertEqual(ctx.extra_options, {"key": "value"})

    def test_default_artifact_dir(self):
        ctx = PipelineContext(
            model_name="test", soc_model=TEST_SOC_MODEL, prompt=["test"]
        )
        self.assertEqual(ctx.artifact_dir, TEST_DEFAULT_ARTIFACT_DIR)

    def test_default_extra_options(self):
        ctx = PipelineContext(
            model_name="test", soc_model=TEST_SOC_MODEL, prompt=["test"]
        )
        self.assertEqual(ctx.extra_options, {})

    def test_frozen(self):
        ctx = PipelineContext(
            model_name="test", soc_model=TEST_SOC_MODEL, prompt=["test"]
        )
        with self.assertRaises(AttributeError):
            ctx.model_name = "changed"

    def test_builder_static_method(self):
        builder = PipelineContext.builder()
        self.assertIsInstance(builder, PipelineContextBuilder)


class TestPipelineContextBuilder(unittest.TestCase):

    def test_build_with_all_required_fields(self):
        ctx = (
            PipelineContext.builder()
            .with_model(TEST_MODEL_NAME)
            .with_soc(TEST_SOC_MODEL)
            .with_prompt(TEST_PROMPT)
            .build()
        )
        self.assertEqual(ctx.model_name, TEST_MODEL_NAME)
        self.assertEqual(ctx.soc_model, TEST_SOC_MODEL)
        self.assertEqual(ctx.prompt, [TEST_PROMPT])

    def test_string_prompt_becomes_list(self):
        ctx = (
            PipelineContext.builder()
            .with_model("test")
            .with_soc(TEST_SOC_MODEL)
            .with_prompt("single prompt")
            .build()
        )
        self.assertEqual(ctx.prompt, ["single prompt"])

    def test_list_prompt_preserved(self):
        test_prompts = ["prompt1", "prompt2"]
        ctx = (
            PipelineContext.builder()
            .with_model("test")
            .with_soc(TEST_SOC_MODEL)
            .with_prompt(test_prompts)
            .build()
        )
        self.assertEqual(ctx.prompt, test_prompts)

    def test_with_artifact_dir(self):
        custom_dir = "/custom/path"
        ctx = (
            PipelineContext.builder()
            .with_model("test")
            .with_soc(TEST_SOC_MODEL)
            .with_prompt("test")
            .with_artifact_dir(custom_dir)
            .build()
        )
        self.assertEqual(ctx.artifact_dir, custom_dir)

    def test_with_extra_options(self):
        extra = {"temperature": 0.8}
        ctx = (
            PipelineContext.builder()
            .with_model("test")
            .with_soc(TEST_SOC_MODEL)
            .with_prompt("test")
            .with_extra_options(extra)
            .build()
        )
        self.assertEqual(ctx.extra_options, extra)

    def test_missing_model_raises(self):
        with self.assertRaises(ValueError) as cm:
            PipelineContext.builder().with_soc(TEST_SOC_MODEL).with_prompt(
                "test"
            ).build()
        self.assertIn("model_name", str(cm.exception))

    def test_missing_soc_raises(self):
        with self.assertRaises(ValueError) as cm:
            PipelineContext.builder().with_model("test").with_prompt("test").build()
        self.assertIn("soc_model", str(cm.exception))

    def test_missing_prompt_raises(self):
        with self.assertRaises(ValueError) as cm:
            PipelineContext.builder().with_model("test").with_soc(
                TEST_SOC_MODEL
            ).build()
        self.assertIn("prompt", str(cm.exception))

    def test_missing_all_raises_with_all_fields(self):
        with self.assertRaises(ValueError) as cm:
            PipelineContext.builder().build()
        msg = str(cm.exception)
        self.assertIn("model_name", msg)
        self.assertIn("soc_model", msg)
        self.assertIn("prompt", msg)

    def test_builder_chaining(self):
        builder = PipelineContextBuilder()
        self.assertIs(builder.with_model("test"), builder)
        self.assertIs(builder.with_soc(TEST_SOC_MODEL), builder)
        self.assertIs(builder.with_prompt("test"), builder)
        self.assertIs(builder.with_artifact_dir("/tmp"), builder)
        self.assertIs(builder.with_extra_options({"k": "v"}), builder)


if __name__ == "__main__":
    unittest.main()
