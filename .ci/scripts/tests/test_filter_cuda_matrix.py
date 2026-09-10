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

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER = _load_module(
    "filter_cuda_matrix", ROOT / ".github" / "scripts" / "filter_cuda_matrix.py"
)
INSTALL_UTILS = _load_module("install_utils", ROOT / "install_utils.py")


def _full_matrix():
    """Every supported pair; TestPublishedSets separately guards against shrinking the lists."""
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

    def test_absent_train_is_skipped_not_fatal(self):
        # A supported train the generator offers nothing for is one PyTorch stopped shipping. The
        # release skips it and publishes the rest, so one dropped train cannot take the others down.
        #
        # Offering every train but the last leaves that train absent while every offered combination
        # stays complete, which is exactly the shape of an upstream drop.
        offered = FILTER.SUPPORTED_CUDA_VERSIONS[:-1]
        dropped = FILTER.SUPPORTED_CUDA_VERSIONS[-1]
        matrix = {
            "include": [
                {"python_version": python, "desired_cuda": cuda}
                for python in FILTER.SUPPORTED_PYTHON_VERSIONS
                for cuda in offered
            ]
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            FILTER.main(["--matrix", json.dumps(matrix)])
        emitted = json.loads(stdout.getvalue())
        self.assertIn("the generator offered no row", stderr.getvalue())
        self.assertIn(dropped, stderr.getvalue())
        published = sorted({row["desired_cuda"] for row in emitted["include"]})
        self.assertEqual(published, sorted(offered))
        self.assertNotIn(dropped, published)

    def test_dropped_train_still_publishes_the_others(self):
        # Losing cu126 from the generator must not block the remaining supported trains.
        if "cu126" not in FILTER.SUPPORTED_CUDA_VERSIONS:
            self.skipTest("cu126 is not a published train")
        survivors = [c for c in FILTER.SUPPORTED_CUDA_VERSIONS if c != "cu126"]
        matrix = {
            "include": [
                {"python_version": python, "desired_cuda": cuda}
                for python in FILTER.SUPPORTED_PYTHON_VERSIONS
                for cuda in survivors
            ]
        }
        emitted = _emitted(_run(matrix))
        published = sorted({row["desired_cuda"] for row in emitted["include"]})
        self.assertEqual(published, sorted(survivors))
        self.assertNotIn("cu126", published)
        # Every survivor keeps all its pythons, so what publishes is complete, just narrower.
        self.assertEqual(
            len(emitted["include"]),
            len(survivors) * len(FILTER.SUPPORTED_PYTHON_VERSIONS),
        )

    def test_missing_combination_exits_nonzero(self):
        # A train that IS offered but missing one python is a real break, not an upstream drop: the
        # release would ship that train incomplete. Deleting one row from a full matrix leaves its
        # train present, so this exercises the incomplete-train gate rather than the skip above.
        matrix = _full_matrix()
        del matrix["include"][0]
        message = self._exit_message(matrix)
        self.assertIn("incomplete train", message)

    def test_offered_train_with_only_unsupported_pythons_exits_nonzero(self):
        for cuda in FILTER.SUPPORTED_CUDA_VERSIONS:
            with self.subTest(cuda=cuda):
                matrix = _full_matrix()
                for row in matrix["include"]:
                    if row["desired_cuda"] == cuda:
                        row["python_version"] = "3.15"
                message = self._exit_message(matrix)
                self.assertIn("incomplete train", message)
                self.assertIn(f"3.10/{cuda}", message)

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


class TestPublishedSets(unittest.TestCase):
    """What a release publishes, pinned against something other than the filter's own lists.

    Every case above builds its fixture from those lists, so shrinking one shrinks the fixture with
    it and every gate still passes. The published set is a promise to users rather than an
    implementation detail, so dropping a row has to be a deliberate edit here too.
    """

    def test_published_cuda_versions(self):
        self.assertEqual(
            FILTER.SUPPORTED_CUDA_VERSIONS, ["cu126", "cu130", "cu132", "cu134"]
        )

    def test_published_cuda_versions_are_supported_by_the_installer(self):
        supported = {
            f"cu{major}{minor}"
            for major, minor in INSTALL_UTILS.SUPPORTED_CUDA_VERSIONS
        }
        self.assertLessEqual(set(FILTER.SUPPORTED_CUDA_VERSIONS), supported)

    def test_supported_toolkits_select_the_matching_torch_index(self):
        base_url = "https://download.pytorch.org/whl/nightly"
        self.addCleanup(INSTALL_UTILS._get_cuda_version.cache_clear)
        self.addCleanup(INSTALL_UTILS.determine_torch_url.cache_clear)
        for major, minor in INSTALL_UTILS.SUPPORTED_CUDA_VERSIONS:
            with self.subTest(cuda=(major, minor)):
                INSTALL_UTILS._get_cuda_version.cache_clear()
                INSTALL_UTILS.determine_torch_url.cache_clear()
                detected = subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=f"Cuda compilation tools, release {major}.{minor}, V{major}.{minor}.0",
                )
                with mock.patch.object(
                    INSTALL_UTILS.platform, "system", return_value="Linux"
                ), mock.patch.object(
                    INSTALL_UTILS.subprocess, "run", return_value=detected
                ):
                    self.assertEqual(
                        INSTALL_UTILS.determine_torch_url(base_url),
                        f"{base_url}/cu{major}{minor}",
                    )
                    self.assertTrue(INSTALL_UTILS.is_cuda_available())

    def test_published_cuda_versions_have_gpu_architectures(self):
        script = ROOT / ".ci" / "scripts" / "wheel" / "cuda_arch_list.sh"
        for machine in ("x86_64", "aarch64"):
            for cuda in FILTER.SUPPORTED_CUDA_VERSIONS:
                with self.subTest(machine=machine, cuda=cuda):
                    result = subprocess.run(
                        [
                            "bash",
                            "-c",
                            'uname() { printf "%s\\n" "$MACHINE"; }; '
                            'source "$1"; executorch_cuda_arch_list',
                            "bash",
                            str(script),
                        ],
                        env={
                            **os.environ,
                            "MACHINE": machine,
                            "CU_VERSION": cuda,
                            "EXECUTORCH_BUILD_CUDA": "1",
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("8.0", result.stdout.split())

    def test_published_python_versions(self):
        self.assertEqual(
            FILTER.SUPPORTED_PYTHON_VERSIONS, ["3.10", "3.11", "3.12", "3.13", "3.14"]
        )

    def test_the_workflows_offer_exactly_the_published_pythons(self):
        # The filter can only keep a row the generator produced, and these two workflows are what
        # tell the generator which pythons to produce. A python published here but not offered
        # there does trip the release gate, but only on a release run, well after the change
        # landed. A python offered there and not published here is dropped without a word.
        for name in (
            "build-wheels-cuda-linux.yml",
            "build-wheels-cuda-aarch64-linux.yml",
        ):
            with self.subTest(workflow=name):
                workflow = yaml.safe_load(
                    (ROOT / ".github" / "workflows" / name).read_text()
                )
                offered = json.loads(
                    workflow["jobs"]["generate-matrix"]["with"]["python-versions"]
                )
                self.assertEqual(offered, FILTER.SUPPORTED_PYTHON_VERSIONS)

    def test_the_pull_request_row_names_a_python_a_pull_request_is_offered(self):
        # A limited pull request is offered one python only, because the shared generator replaces
        # the list the workflow passes with its first entry. Naming any other one here matched no
        # offered row, so the row a pull request built was not the row this file names.
        offered = {
            "include": [
                {
                    "python_version": FILTER.SUPPORTED_PYTHON_VERSIONS[0],
                    "desired_cuda": cuda,
                }
                for cuda in FILTER.SUPPORTED_CUDA_VERSIONS
            ]
        }
        emitted = _emitted(_run(offered, limit="true"))
        self.assertEqual(
            emitted["include"],
            [
                {
                    "python_version": FILTER.PR_PYTHON_VERSION,
                    "desired_cuda": FILTER.PR_CUDA_VERSION,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
