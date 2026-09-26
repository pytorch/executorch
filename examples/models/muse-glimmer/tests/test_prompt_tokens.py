# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Golden prompt-token tests, the invariance guard for the tokenizer refactor.

Three tiers, each running only when its inputs are available:

1. Structure. No assets. Asserts the prompt skeleton: BOS first, image span
   shape, ordering. Always runs.
2. Real tokenizer. Needs ``MUSE_GLIMMER_HF_DIR`` pointing at a directory with
   ``tokenizer.json``. Asserts token counts and a digest of the full id
   sequence against tests/data/prompt_golden.json.
3. Runner. Runs a binary with ``--max_new_tokens 0`` and asserts the
   ``Prompt tokens:`` line matches the golden count. Four combinations, each
   gated on its own variables and skipped, never failed, when they are unset:

     solo   text    MUSE_GLIMMER_SOLO_RUNNER   + MUSE_GLIMMER_SOLO_PTE
     solo   image   ... + MUSE_GLIMMER_SOLO_VISION_PTE
     dflash text    MUSE_GLIMMER_DFLASH_RUNNER + MUSE_GLIMMER_DFLASH_PTE
     dflash image   ... + MUSE_GLIMMER_DFLASH_VISION_PTE

   All four also need ``MUSE_GLIMMER_HF_DIR``. The image combinations use
   ``MUSE_GLIMMER_TEST_IMAGE`` when set and otherwise synthesize one, sizing the
   expected patch count with ``vision.precompute.compute_grid_size``. A sibling
   ``.ptd`` is passed as ``--data_path`` automatically, which CUDA exports need.

   Prompt assembly is duplicated per binary (solo.cpp:553, dflash.cpp:417) and
   the image span has two independent validators (muse_glimmer_engine.cpp:716,
   dflash_session.cpp:506), so one binary does not stand in for the other.

   Known gap: this tier compares counts, not ids. A same-length substitution,
   say a wrong BOS id, passes here and is caught only by tier 2, which exercises
   the Python tokenizer rather than the runner's. Closing it needs the runner to
   be able to dump its prompt ids. Nothing here observes the EOS set either.

Usage:
    python dev/run.py unittest -v \
      executorch.examples.models.muse_glimmer.tests.test_prompt_tokens

    MUSE_GLIMMER_HF_DIR=assets/hf python dev/run.py unittest -v \
      executorch.examples.models.muse_glimmer.tests.test_prompt_tokens

A skip still exits 0, so read the "Ran N tests" block, not the exit code.
"""

import json
import os
import re
import subprocess
import tempfile
import unittest

from executorch.examples.models.muse_glimmer.tests import prompt_cases as pc

GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "prompt_golden.json"
)
PROMPT_TOKENS_RE = re.compile(r"^Prompt tokens:\s*(\d+)\s*$", re.MULTILINE)


def load_golden():
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


class PromptStructure(unittest.TestCase):
    """Tier 1: shape of the assembled prompt, no assets required."""

    def ids(self, case):
        return pc.build_prompt_ids(pc.stub_encode, case)

    def test_bos_is_first_and_appears_once(self):
        for case in pc.CASES:
            with self.subTest(case=case.name):
                ids = self.ids(case)
                self.assertEqual(ids[0], pc.BOS_ID)
                self.assertEqual(ids.count(pc.BOS_ID), 1)

    def test_text_only_cases_have_no_image_tokens(self):
        for case in (c for c in pc.CASES if not c.num_soft_tokens):
            with self.subTest(case=case.name):
                ids = self.ids(case)
                for tok in (pc.IMAGE_START_ID, pc.IMAGE_END_ID, pc.PATCH_ID):
                    self.assertNotIn(tok, ids)

    def test_image_span_shape(self):
        """Canonical contract: one bare patch run replacing ``<img>``."""
        for case in (c for c in pc.CASES if c.num_soft_tokens):
            with self.subTest(case=case.name):
                ids = self.ids(case)
                n = case.num_soft_tokens
                self.assertEqual(ids.count(pc.PATCH_ID), n)
                for wrapper in (pc.IMAGE_START_ID, pc.IMAGE_END_ID):
                    self.assertNotIn(wrapper, ids)

    def test_image_span_replaces_marker_in_place(self):
        case = next(c for c in pc.CASES if c.num_soft_tokens)
        ids = self.ids(case)
        before, after = case.prompt.split(pc.IMAGE_MARKER)
        patch_start = 1 + len(pc.stub_encode(before))
        patch_end = patch_start + case.num_soft_tokens
        self.assertEqual(
            ids,
            [pc.BOS_ID]
            + pc.stub_encode(before)
            + [pc.PATCH_ID] * case.num_soft_tokens
            + pc.stub_encode(after),
        )
        self.assertEqual(
            ids[patch_start:patch_end], [pc.PATCH_ID] * case.num_soft_tokens
        )

    def test_image_costs_exactly_its_patches(self):
        """No wrapper: an image costs its patch run and nothing else."""
        for with_image in (c for c in pc.CASES if c.num_soft_tokens):
            with self.subTest(case=with_image.name):
                text_only = pc.Case(
                    "bare", with_image.prompt.replace(pc.IMAGE_MARKER, "")
                )
                self.assertEqual(
                    len(self.ids(with_image)) - len(self.ids(text_only)),
                    with_image.num_soft_tokens + pc.IMAGE_WRAPPER_TOKENS,
                )

    def test_case_names_are_unique(self):
        names = [c.name for c in pc.CASES]
        self.assertEqual(len(names), len(set(names)))


class RealTokenizerGolden(unittest.TestCase):
    """Tier 2: exact ids against the committed golden."""

    @classmethod
    def setUpClass(cls):
        cls.dir = pc.hf_dir()
        if cls.dir is None:
            raise unittest.SkipTest(f"{pc.HF_DIR_ENV} unset or has no tokenizer.json")
        cls.encode = staticmethod(pc.real_encoder(cls.dir))
        cls.golden = load_golden()

    def test_tokenizer_matches_the_one_the_golden_was_built_from(self):
        actual = pc.file_sha256(os.path.join(self.dir, "tokenizer.json"))
        self.assertEqual(
            actual,
            self.golden["tokenizer_sha256"],
            "tokenizer.json differs from the one used to generate the golden; "
            "regenerate with tests/gen_prompt_golden.py and explain the diff",
        )

    def test_every_case_is_covered(self):
        self.assertEqual(sorted(self.golden["cases"]), sorted(c.name for c in pc.CASES))

    def test_token_counts_and_ids_match_golden(self):
        for case in pc.CASES:
            with self.subTest(case=case.name):
                want = self.golden["cases"][case.name]
                ids = pc.build_prompt_ids(self.encode, case)
                self.assertEqual(len(ids), want["num_tokens"])
                self.assertEqual(pc.special_runs(ids), want["special_runs"])
                self.assertEqual(pc.ids_sha256(ids), want["ids_sha256"])

    def test_harmony_pieces_resolve_to_the_documented_ids(self):
        """Guards against a tokenizer that silently BPEs the control tokens."""
        expected = {
            "<|begin_of_text|>": 200000,
            "<|end_of_text|>": 200001,
            "<|eom|>": 200007,
            "<|eot|>": 200008,
            "<|start|>": 200022,
            "<|message|>": 200023,
            "<|image_start|>": 200080,
            "<|image_end|>": 200081,
            "<|patch|>": 200092,
        }
        for piece, want in expected.items():
            with self.subTest(piece=piece):
                self.assertEqual(self.encode(piece), [want])

    def test_end_token_is_not_in_the_vocabulary(self):
        """<|end|> looks plausible and is not real; it BPEs into ordinary text."""
        self.assertGreater(len(self.encode("<|end|>")), 1)

    def readme_text(self):
        readme = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "README.md"
        )
        with open(readme) as fh:
            return fh.read()

    def test_every_control_token_in_the_readme_is_real(self):
        """A doc example using a non-existent token is wrong but not rejected."""
        pieces = set(re.findall(r"<\|[a-z0-9_]+\|>", self.readme_text()))
        pieces.discard("<|end|>")  # covered by the test below, which is stricter
        self.assertTrue(pieces, "no control tokens found; did the README move?")
        for piece in sorted(pieces):
            with self.subTest(piece=piece):
                self.assertEqual(
                    len(self.encode(piece)),
                    1,
                    f"{piece} is not a single token; it will encode as text",
                )

    def test_end_appears_only_in_the_sentence_denying_it_exists(self):
        """Blanket-excluding <|end|> would let an example reintroduce it unseen.

        Pin the one legitimate mention instead, so a second occurrence anywhere,
        prose or example, fails.
        """
        text = self.readme_text()
        occurrences = [m.start() for m in re.finditer(re.escape("<|end|>"), text)]
        self.assertEqual(
            len(occurrences),
            1,
            f"<|end|> is not a real token; expected exactly one mention "
            f"explaining that, found {len(occurrences)}",
        )
        self.assertIn(
            "There is no\n`<|end|>` token",
            text,
            "the single <|end|> is no longer the sentence denying it exists",
        )


class RunnerPromptCount(unittest.TestCase):
    """Tier 3: the C++ path, checked through the printed prompt-token count."""

    RUNNER_TIMEOUT_S = 900
    SYNTHETIC_IMAGE_SIZE = (224, 224)

    def target(self, runner_env, pte_env):
        """Resolve one runner/model pair, or skip listing precisely what is absent."""
        reasons = []
        directory = pc.hf_dir()
        if directory is None:
            reasons.append(f"{pc.HF_DIR_ENV} unset or has no tokenizer.json")
        resolved = {}
        for env in (runner_env, pte_env):
            value = os.environ.get(env)
            if not value:
                reasons.append(f"{env} unset")
            elif not os.path.isfile(os.path.expanduser(value)):
                reasons.append(f"{env}={value} not found")
            else:
                resolved[env] = os.path.expanduser(value)
        if reasons:
            self.skipTest("; ".join(reasons))
        return resolved[runner_env], resolved[pte_env], directory

    @staticmethod
    def data_path_for(pte):
        """CUDA exports put delegate weights in a sibling .ptd the runner needs."""
        directory = os.path.dirname(pte) or "."
        ptds = sorted(f for f in os.listdir(directory) if f.endswith(".ptd"))
        return os.path.join(directory, ptds[0]) if len(ptds) == 1 else None

    def prompt_tokens(self, runner, pte, directory, prompt, image=None):
        cmd = [
            runner,
            "--model_path",
            pte,
            "--tokenizer_path",
            os.path.join(directory, "tokenizer.json"),
            "--prompt",
            prompt,
            # Not 0: dflash_runner rejects a non-positive value in validate_flags
            # and exits before tokenizing, and it prints the count only after the
            # decode loop. One token also means prefill really runs, so the image
            # span validators are exercised rather than skipped.
            "--max_new_tokens",
            "1",
        ]
        data_path = self.data_path_for(pte)
        if data_path:
            cmd += ["--data_path", data_path]
        if image:
            cmd += ["--image_path", image]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.RUNNER_TIMEOUT_S
        )
        context = (
            f"{os.path.basename(runner)} exit={proc.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
        )
        # solo_runner prints the count before prefill, so a later failure would
        # otherwise be read as a pass.
        self.assertEqual(proc.returncode, 0, f"runner failed.\n{context}")
        match = PROMPT_TOKENS_RE.search(proc.stdout)
        self.assertIsNotNone(match, f"no 'Prompt tokens:' line.\n{context}")
        return int(match.group(1))

    def image_and_patch_count(self):
        """Path to an image plus the soft-token count the vision encoder will emit."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        from executorch.examples.models.muse_glimmer.vision.precompute import (
            compute_grid_size,
        )

        supplied = os.environ.get("MUSE_GLIMMER_TEST_IMAGE")
        if supplied:
            path = os.path.expanduser(supplied)
            if not os.path.isfile(path):
                self.skipTest(f"MUSE_GLIMMER_TEST_IMAGE={supplied} not found")
            with Image.open(path) as img:
                width, height = img.size
        else:
            width, height = self.SYNTHETIC_IMAGE_SIZE
            handle, path = tempfile.mkstemp(suffix=".png")
            os.close(handle)
            self.addCleanup(os.unlink, path)
            Image.new("RGB", (width, height), (127, 127, 127)).save(path)
        return path, compute_grid_size(width, height)[2]

    def check_text_cases(self, runner_env, pte_env):
        runner, pte, directory = self.target(runner_env, pte_env)
        golden = load_golden()
        for case in (c for c in pc.CASES if not c.num_soft_tokens):
            with self.subTest(case=case.name):
                self.assertEqual(
                    self.prompt_tokens(runner, pte, directory, case.prompt),
                    golden["cases"][case.name]["num_tokens"],
                )

    def check_image_case(self, runner_env, pte_env):
        runner, pte, directory = self.target(runner_env, pte_env)
        golden = load_golden()
        case = next(c for c in pc.CASES if c.num_soft_tokens)
        image, num_soft_tokens = self.image_and_patch_count()
        expected = (
            pc.text_only_len(golden["cases"][case.name], case)
            + num_soft_tokens
            + pc.IMAGE_WRAPPER_TOKENS
        )
        self.assertEqual(
            self.prompt_tokens(runner, pte, directory, case.prompt, image=image),
            expected,
            f"expected text({pc.text_only_len(golden['cases'][case.name], case)}) "
            f"+ patches({num_soft_tokens}) + wrapper({pc.IMAGE_WRAPPER_TOKENS})",
        )

    def test_solo_text(self):
        self.check_text_cases("MUSE_GLIMMER_SOLO_RUNNER", "MUSE_GLIMMER_SOLO_PTE")

    def test_solo_image(self):
        self.check_image_case(
            "MUSE_GLIMMER_SOLO_RUNNER", "MUSE_GLIMMER_SOLO_VISION_PTE"
        )

    def test_dflash_text(self):
        self.check_text_cases("MUSE_GLIMMER_DFLASH_RUNNER", "MUSE_GLIMMER_DFLASH_PTE")

    def test_dflash_image(self):
        self.check_image_case(
            "MUSE_GLIMMER_DFLASH_RUNNER", "MUSE_GLIMMER_DFLASH_VISION_PTE"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
