# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for the CUDA release matrix filter.
#
# The filter decides which wheel rows a release builds and exits non-zero when its inputs disagree
# with what the project can publish. Two of its comments record past bugs it now guards against, and
# a regression in any of them would surface only as a broken release, so each gate is pinned here.

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


def _load_filter():
    """Load the script by path, since .github/scripts is not an importable package."""
    root = Path(__file__).resolve().parents[3]
    path = root / ".github" / "scripts" / "filter_cuda_matrix.py"
    spec = importlib.util.spec_from_file_location("filter_cuda_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER = _load_filter()


def _full_matrix():
    """Every supported python and CUDA pair.

    The filter refuses anything less: one gate rejects a matrix that would leave a CUDA train
    unpublished, another rejects a missing python and CUDA combination. Built from the module's own
    lists so it cannot go stale when either grows.
    """
    return {
        "include": [
            {"python_version": python, "desired_cuda": cuda}
            for python in FILTER.SUPPORTED_PYTHON_VERSIONS
            for cuda in FILTER.SUPPORTED_CUDA_VERSIONS
        ]
    }


def _run(matrix, limit="false", extra=None):
    argv = ["--matrix", json.dumps(matrix), "--limit-pr-builds", limit] + (extra or [])
    with mock.patch("builtins.print") as printed:
        FILTER.main(argv)
    return printed


def _emitted(printed):
    return json.loads(printed.call_args_list[-1].args[0])


class TestRanking(unittest.TestCase):
    def test_prefers_the_requested_cuda_over_a_newer_one(self):
        # The ranking deliberately scores a version above the requested one NEGATIVELY, so a newer
        # one never outranks the one a machine here can actually run. A fixture offering only
        # versions at or below the request never executes that branch.
        newer = [
            c for c in FILTER.SUPPORTED_CUDA_VERSIONS if c > FILTER.PR_CUDA_VERSION
        ]
        items = [
            {
                "python_version": FILTER.PR_PYTHON_VERSION,
                "desired_cuda": FILTER.PR_CUDA_VERSION,
            }
        ] + [
            {"python_version": FILTER.PR_PYTHON_VERSION, "desired_cuda": c}
            for c in newer
        ]
        picked = FILTER.only_pull_request_row(items)
        self.assertEqual(picked[0]["desired_cuda"], FILTER.PR_CUDA_VERSION)

    def test_cuda_closeness_outranks_the_python_match(self):
        # Closeness is the FIRST element of the sort key, deliberately. Ranking python first is a
        # recorded past bug: it picked a wheel for a CUDA version nothing on hand can execute.
        other_python = next(
            p for p in FILTER.SUPPORTED_PYTHON_VERSIONS if p != FILTER.PR_PYTHON_VERSION
        )
        other_cuda = next(
            c for c in FILTER.SUPPORTED_CUDA_VERSIONS if c != FILTER.PR_CUDA_VERSION
        )
        items = [
            {"python_version": other_python, "desired_cuda": FILTER.PR_CUDA_VERSION},
            {"python_version": FILTER.PR_PYTHON_VERSION, "desired_cuda": other_cuda},
        ]
        picked = FILTER.only_pull_request_row(items)
        self.assertEqual(picked[0]["desired_cuda"], FILTER.PR_CUDA_VERSION)

    def test_picks_the_requested_row(self):
        items = [
            {"python_version": p, "desired_cuda": c}
            for p in FILTER.SUPPORTED_PYTHON_VERSIONS
            for c in FILTER.SUPPORTED_CUDA_VERSIONS
        ]
        picked = FILTER.only_pull_request_row(items)
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0]["python_version"], FILTER.PR_PYTHON_VERSION)
        self.assertEqual(picked[0]["desired_cuda"], FILTER.PR_CUDA_VERSION)

    def test_empty_input_gives_empty_output(self):
        # Raising here would break every pull request while releases kept working, which is one of
        # the two failures this function records having had.
        self.assertEqual(FILTER.only_pull_request_row([]), [])

    def test_requested_cuda_absent_from_the_offer(self):
        # The other recorded past bug: the requested version falls off the supported list, and the
        # function still has to return one row rather than raise or return nothing.
        items = [
            {"python_version": FILTER.PR_PYTHON_VERSION, "desired_cuda": c}
            for c in FILTER.SUPPORTED_CUDA_VERSIONS
            if c != FILTER.PR_CUDA_VERSION
        ]
        picked = FILTER.only_pull_request_row(items)
        self.assertEqual(len(picked), 1)


class TestVersionRank(unittest.TestCase):
    def test_newer_cuda_ranks_higher(self):
        ordered = sorted(FILTER.SUPPORTED_CUDA_VERSIONS)
        self.assertGreater(
            FILTER._version_rank(ordered[-1]), FILTER._version_rank(ordered[0])
        )

    def test_unknown_value_ranks_below_every_real_one(self):
        # A value ranking above the real ones would silently take over the pull request row.
        self.assertEqual(FILTER._version_rank("not-a-version"), -1)


class TestKeep(unittest.TestCase):
    def test_unsupported_python_is_dropped(self):
        # The recorded bug: passing a 3.9 row returned success and emitted it.
        matrix = _full_matrix()
        matrix["include"].append(
            {"python_version": "3.9", "desired_cuda": FILTER.SUPPORTED_CUDA_VERSIONS[0]}
        )
        emitted = _emitted(_run(matrix))
        self.assertNotIn("3.9", [row["python_version"] for row in emitted["include"]])

    def test_unsupported_cuda_is_dropped(self):
        matrix = _full_matrix()
        matrix["include"].append(
            {
                "python_version": FILTER.SUPPORTED_PYTHON_VERSIONS[0],
                "desired_cuda": "cu999",
            }
        )
        emitted = _emitted(_run(matrix))
        self.assertNotIn("cu999", [row["desired_cuda"] for row in emitted["include"]])


class TestGates(unittest.TestCase):
    def _exit_message(self, matrix, limit="false", extra=None):
        """The stderr text of the gate that fired, so a case can name which one it hit."""
        import contextlib
        import io

        argv = ["--matrix", json.dumps(matrix), "--limit-pr-builds", limit] + (
            extra or []
        )
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            with self.assertRaises(SystemExit) as raised:
                FILTER.main(argv)
        self.assertNotEqual(raised.exception.code, 0)
        return captured.getvalue()

    def _expect_exit(self, matrix, limit="false", extra=None):
        with mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as raised:
                _run(matrix, limit=limit, extra=extra)
        self.assertNotEqual(raised.exception.code, 0)

    def test_unparseable_matrix_exits_nonzero(self):
        argv = ["--matrix", "{not json", "--limit-pr-builds", "false"]
        with mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as raised:
                FILTER.main(argv)
        self.assertNotEqual(raised.exception.code, 0)

    def test_absent_train_exits_nonzero(self):
        # A supported train the generator offers nothing for would publish no wheel at all.
        #
        # Patching the supported list rather than deleting rows, because deleting every row for one
        # train also creates missing combinations, so both gates fire and the test cannot tell which
        # one it exercised. Adding an extra supported train makes it absent while every offered
        # combination stays complete.
        # These two gates cannot be separated by input: any matrix leaving a train absent also
        # leaves every combination for that train missing, so the later gate always catches what the
        # earlier one would. Measured. So each gate gets its own case, and the case asserts on the
        # message rather than only on a nonzero exit, which is the only way to tell them apart.
        offered = FILTER.SUPPORTED_CUDA_VERSIONS[:-1]
        matrix = {
            "include": [
                {"python_version": python, "desired_cuda": cuda}
                for python in FILTER.SUPPORTED_PYTHON_VERSIONS
                for cuda in offered
            ]
        }
        message = self._exit_message(matrix)
        self.assertIn("publish no wheel for that CUDA version", message)

    def test_missing_combination_exits_nonzero(self):
        matrix = _full_matrix()
        del matrix["include"][0]
        message = self._exit_message(matrix)
        self.assertIn("combination(s) produced no row", message)

    def test_jetpack_not_published_exits_nonzero(self):
        # Refused explicitly rather than allowed to fall through to an empty result, so the reason a
        # reader sees is the real one. Nothing passes this flag today, which is why it had no cover.
        message = self._exit_message(_full_matrix(), extra=["--jetpack", "true"])
        self.assertIn("JetPack rows are not published yet", message)

    def test_empty_result_exits_nonzero(self):
        self._expect_exit({"include": []})

    def test_well_formed_matrix_passes_through(self):
        matrix = _full_matrix()
        emitted = _emitted(_run(matrix))
        self.assertEqual(emitted["include"], matrix["include"])

    def test_pull_request_limit_reduces_to_one_row(self):
        emitted = _emitted(_run(_full_matrix(), limit="true"))
        self.assertEqual(len(emitted["include"]), 1)


if __name__ == "__main__":
    unittest.main()
