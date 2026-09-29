# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate tests/data/prompt_golden.json from the real tokenizer.

    MUSE_GLIMMER_HF_DIR=assets/hf \
      python dev/run.py executorch.examples.models.muse_glimmer.tests.gen_prompt_golden

Only lengths, the special-token skeleton, and a digest of the full id sequence
are written. The text ids themselves stay out of the repo.
"""

import json
import os
import sys

from executorch.examples.models.muse_glimmer.tests import prompt_cases as pc

GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "prompt_golden.json"
)


def build():
    directory = pc.hf_dir()
    if directory is None:
        sys.exit(f"Set {pc.HF_DIR_ENV} to a directory containing tokenizer.json")
    encode = pc.real_encoder(directory)

    entries = {}
    for case in pc.CASES:
        ids = pc.build_prompt_ids(encode, case)
        entries[case.name] = {
            "prompt": case.prompt,
            "num_soft_tokens": case.num_soft_tokens,
            "num_tokens": len(ids),
            "special_runs": pc.special_runs(ids),
            "ids_sha256": pc.ids_sha256(ids),
        }
    return {
        "comment": "Regenerate with tests/gen_prompt_golden.py. "
        "Commits that change these numbers must say why.",
        "tokenizer_sha256": pc.file_sha256(os.path.join(directory, "tokenizer.json")),
        "cases": entries,
    }


if __name__ == "__main__":
    golden = build()
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(golden, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN_PATH}")
    for name, e in sorted(golden["cases"].items()):
        print(f"  {name:<32} {e['num_tokens']:>5} tokens  {e['ids_sha256'][:16]}")
