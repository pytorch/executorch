# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from dataclasses import fields

from executorch.backends.qualcomm.genai_pipeline.control_args import (
    _PARSER_DEFAULT_EXEMPT_FIELDS,
    ControlArgs,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext

# Arguments llama.py's parser marks as required, so it can be run with no
# user-facing options and still produce a namespace full of defaults.
_REQUIRED_PARSER_ARGS = ["--decoder_model", "stories260k", "--prompt", "hi"]


def _make_context(**overrides):
    """Create a PipelineContext with valid defaults."""
    defaults = {
        "model_name": "stories260k",
        "soc_model": "SM8750",
        "prompt": ["hello"],
        "artifact_dir": "/tmp/artifacts",
        "extra_options": {},
    }
    defaults.update(overrides)
    return PipelineContext(**defaults)


class TestControlArgsDefaults(unittest.TestCase):

    def test_defaults_match_llama_parser(self):
        """Every shared field defaults to the same value as llama.py's parser."""
        from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser

        parsed = vars(_build_parser().parse_args(_REQUIRED_PARSER_ARGS))
        ours = ControlArgs()

        shared = (
            ControlArgs.field_names() & set(parsed)
        ) - _PARSER_DEFAULT_EXEMPT_FIELDS
        # The parser supplies these two, so they are not defaults to compare.
        shared -= {"decoder_model", "prompt"}

        for name in sorted(shared):
            with self.subTest(field=name):
                self.assertEqual(getattr(ours, name), parsed[name])

    def test_every_field_is_known_to_the_parser(self):
        """No field exists that llama.py's parser cannot supply."""
        from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser

        parsed = vars(_build_parser().parse_args(_REQUIRED_PARSER_ARGS))

        self.assertEqual(ControlArgs.field_names() - set(parsed), set())

    def test_exempt_fields_default_to_none_where_the_parser_has_a_path(self):
        """The exempted fields differ from the parser deliberately, not by drift.

        llama.py defaults them to YAML paths under examples/, which this package
        does not reference; None means "use the consumer's own default".
        """
        from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser

        parsed = vars(_build_parser().parse_args(_REQUIRED_PARSER_ARGS))
        ours = ControlArgs()

        for name in sorted(_PARSER_DEFAULT_EXEMPT_FIELDS):
            with self.subTest(field=name):
                self.assertIsNone(getattr(ours, name))
                self.assertIsNotNone(parsed[name])

    def test_is_argparse_namespace(self):
        """ControlArgs is a Namespace, as QnnConfig.load_config requires."""
        self.assertIsInstance(ControlArgs(), argparse.Namespace)

    def test_vars_exposes_all_fields(self):
        """vars() yields every field, so reflection-based consumers see them."""
        self.assertEqual(
            set(vars(ControlArgs())),
            {f.name for f in fields(ControlArgs)},
        )

    def test_mutable_defaults_are_not_shared(self):
        """Each instance gets its own list defaults."""
        first, second = ControlArgs(), ControlArgs()

        first.prompt.append("mutated")

        self.assertEqual(second.prompt, [])


class TestControlArgsFromNamespace(unittest.TestCase):

    def test_copies_recognised_attributes(self):
        """Attributes matching a field are carried over."""
        namespace = argparse.Namespace(decoder_model="qwen3-0_6b", max_seq_len=2048)

        result = ControlArgs.from_namespace(namespace)

        self.assertEqual(result.decoder_model, "qwen3-0_6b")
        self.assertEqual(result.max_seq_len, 2048)

    def test_ignores_unrecognised_attributes(self):
        """Attributes without a field are dropped rather than raising."""
        namespace = argparse.Namespace(decoder_model="x", not_a_field=object())

        result = ControlArgs.from_namespace(namespace)

        self.assertFalse(hasattr(result, "not_a_field"))

    def test_unsupplied_attributes_keep_defaults(self):
        """Fields absent from the namespace fall back to their defaults."""
        result = ControlArgs.from_namespace(argparse.Namespace(decoder_model="x"))

        self.assertEqual(result.model_mode, "hybrid")

    def test_accepts_the_real_llama_parser_namespace(self):
        """A namespace from llama.py's parser converts without error."""
        from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser

        namespace = _build_parser().parse_args(_REQUIRED_PARSER_ARGS)

        result = ControlArgs.from_namespace(namespace)

        self.assertEqual(result.decoder_model, "stories260k")
        self.assertEqual(result.prompt, ["hi"])


class TestControlArgsFromPipelineContext(unittest.TestCase):

    def test_maps_context_owned_fields(self):
        """model_name, soc_model, artifact_dir and prompt come from the context."""
        context = _make_context(
            model_name="llama3_2-1b_instruct",
            soc_model="SM8650",
            artifact_dir="/tmp/out",
            prompt=["a", "b"],
        )

        result = ControlArgs.from_pipeline_context(context)

        self.assertEqual(result.decoder_model, "llama3_2-1b_instruct")
        self.assertEqual(result.soc_model, "SM8650")
        self.assertEqual(result.artifact, "/tmp/out")
        self.assertEqual(result.prompt, ["a", "b"])

    def test_applies_matching_extra_options(self):
        """extra_options keys naming a field are applied."""
        context = _make_context(extra_options={"max_seq_len": 1024, "use_fp16": True})

        result = ControlArgs.from_pipeline_context(context)

        self.assertEqual(result.max_seq_len, 1024)
        self.assertTrue(result.use_fp16)

    def test_ignores_unrelated_extra_options(self):
        """extra_options keys that are not fields are skipped silently."""
        context = _make_context(extra_options={"generate_etrecord": True})

        result = ControlArgs.from_pipeline_context(context)

        self.assertFalse(hasattr(result, "generate_etrecord"))

    def test_overrides_take_precedence_over_extra_options(self):
        """Explicit overrides beat extra_options and context values."""
        context = _make_context(extra_options={"max_seq_len": 1024})

        result = ControlArgs.from_pipeline_context(context, max_seq_len=256)

        self.assertEqual(result.max_seq_len, 256)

    def test_extra_options_cannot_replace_context_owned_fields(self):
        """The context wins over extra_options for the settings it owns.

        A SoC arriving via extra_options would otherwise diverge from the one the
        context reports and the stage configs use, compiling for one target while
        validating ops for another.
        """
        context = _make_context(
            extra_options={
                "soc_model": "SM8650",
                "decoder_model": "other_model",
                "artifact": "/tmp/elsewhere",
                "prompt": ["injected"],
            },
        )

        result = ControlArgs.from_pipeline_context(context)

        self.assertEqual(result.soc_model, context.soc_model)
        self.assertEqual(result.decoder_model, context.model_name)
        self.assertEqual(result.artifact, context.artifact_dir)
        self.assertEqual(result.prompt, context.prompt)

    def test_explicit_override_still_replaces_a_context_owned_field(self):
        """Overrides remain the deliberate way to change a context-owned field."""
        context = _make_context(extra_options={"soc_model": "SM8650"})

        result = ControlArgs.from_pipeline_context(context, soc_model="SM8450")

        self.assertEqual(result.soc_model, "SM8450")

    def test_rejects_unknown_override(self):
        """An override that is not a field raises rather than being dropped."""
        context = _make_context()

        with self.assertRaises(TypeError) as cm:
            ControlArgs.from_pipeline_context(context, not_a_field=1)

        self.assertIn("not_a_field", str(cm.exception))

    def test_prompt_is_copied_from_the_context(self):
        """Mutating the result's prompt does not affect the context."""
        context = _make_context(prompt=["original"])

        result = ControlArgs.from_pipeline_context(context)
        result.prompt.append("added")

        self.assertEqual(context.prompt, ["original"])


class TestControlArgsBuildParser(unittest.TestCase):

    def test_parsing_no_arguments_reproduces_the_dataclass_defaults(self):
        """The parser is the same defaults, so an empty command line matches."""
        parsed = ControlArgs.from_namespace(ControlArgs.build_parser().parse_args([]))

        self.assertEqual(parsed, ControlArgs())

    def test_values_are_converted_to_their_field_type(self):
        """Numeric fields arrive as numbers, not the strings argparse read."""
        args = ControlArgs.build_parser().parse_args(
            ["--max-seq-len", "1024", "--temperature", "0.5"]
        )

        self.assertEqual(args.max_seq_len, 1024)
        self.assertEqual(args.temperature, 0.5)

    def test_underscore_and_hyphen_spellings_both_work(self):
        """``llama.py`` uses both conventions, so accept either."""
        parser = ControlArgs.build_parser()

        self.assertEqual(parser.parse_args(["--max-seq-len", "8"]).max_seq_len, 8)
        self.assertEqual(parser.parse_args(["--max_seq_len", "8"]).max_seq_len, 8)

    def test_list_valued_fields_accept_several_values(self):
        """Prompts and task lists are repeated arguments, not one string."""
        args = ControlArgs.build_parser().parse_args(["--prompt", "one", "two"])

        self.assertEqual(args.prompt, ["one", "two"])

    def test_constrained_fields_reject_an_unlisted_value(self):
        """A misspelled mode fails at parse time, not deep in the flow."""
        parser = ControlArgs.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--model-mode", "not_a_mode"])

    def test_optional_numeric_fields_parse_as_numbers(self):
        """A field defaulting to None still converts, since its type is declared.

        ``llama.py`` uses ``max_context_len`` in arithmetic and comparisons, so a
        string would silently compare wrong rather than fail.
        """
        args = ControlArgs.build_parser().parse_args(
            ["--max-context-len", "1024", "--seed", "7", "--eval-num-fewshot", "3"]
        )

        self.assertEqual(args.max_context_len, 1024)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.eval_num_fewshot, 3)

    def test_parsed_values_match_the_llama_parser(self):
        """Both parsers produce the same values, not merely the same defaults."""
        from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser

        argv = [
            "--max_context_len",
            "1024",
            "--calib_num_fewshot",
            "2",
            "--eval_num_fewshot",
            "3",
            "--seed",
            "7",
            "--max_seq_len",
            "256",
            "--temperature",
            "0.5",
            "--artifact",
            "/tmp/out",
        ]
        theirs = vars(_build_parser().parse_args(argv + _REQUIRED_PARSER_ARGS))
        ours = vars(ControlArgs.build_parser().parse_args(argv))

        # The exempt fields differ by design; the parser supplies the other two.
        shared = (set(ours) & set(theirs)) - _PARSER_DEFAULT_EXEMPT_FIELDS
        shared -= {"decoder_model", "prompt"}

        for name in sorted(shared):
            with self.subTest(field=name):
                self.assertEqual(ours[name], theirs[name])

    def test_parser_can_be_extended_as_a_parent(self):
        """Entry points add their own arguments rather than redeclaring these."""
        parent = ControlArgs.build_parser(add_help=False)
        parser = argparse.ArgumentParser(parents=[parent])
        parser.add_argument("--list-models", action="store_true")

        args = parser.parse_args(["--list-models", "--backend", "gpu"])

        self.assertTrue(args.list_models)
        self.assertEqual(args.backend, "gpu")


if __name__ == "__main__":
    unittest.main()
