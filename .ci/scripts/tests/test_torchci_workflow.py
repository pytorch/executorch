# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"
REUSABLE = "./.github/workflows/_unittest.yml"


def load_workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


class TorchciWorkflowTest(unittest.TestCase):
    def test_reporting_is_off_by_default_and_linux_only(self):
        workflow = load_workflow("_unittest.yml")
        triggers = workflow.get("on", workflow.get(True))
        report_input = triggers["workflow_call"]["inputs"]["upload-test-run-reports"]
        self.assertEqual(report_input["type"], "boolean")
        self.assertIs(report_input["default"], False)
        jobs = workflow["jobs"]
        self.assertEqual(
            jobs["linux"]["with"]["upload-test-run-reports"],
            "${{ inputs.upload-test-run-reports && inputs.build-tool == 'cmake' "
            "&& inputs.editable != 'true' }}",
        )
        for platform in ("macos", "windows"):
            self.assertNotIn("upload-test-run-reports", jobs[platform]["with"])
        for job in load_workflow("_test_backend.yml")["jobs"].values():
            self.assertNotIn("upload-test-run-reports", job.get("with", {}))

    def test_linux_does_not_lower_the_callers_actions_permission(self):
        workflow = load_workflow("_unittest.yml")
        self.assertNotIn("permissions", workflow)
        self.assertNotIn("permissions", workflow["jobs"]["linux"])
        self.assertEqual(
            workflow["jobs"]["linux"]["uses"],
            "pytorch/test-infra/.github/workflows/linux_job_v3.yml@main",
        )

    def test_all_four_callers_have_the_intended_opt_in_and_permission(self):
        callers = {}
        for path in WORKFLOWS.glob("*.yml"):
            for name, job in load_workflow(path.name).get("jobs", {}).items():
                if job.get("uses") == REUSABLE:
                    callers[(path.name, name)] = job
        selected = {("pull.yml", "unittest"), ("trunk.yml", "unittest-release")}
        self.assertEqual(
            set(callers),
            selected
            | {("pull.yml", "unittest-editable"), ("pull.yml", "unittest-buck")},
        )
        for key, job in callers.items():
            with self.subTest(caller=key):
                opted_in = key in selected
                self.assertEqual(
                    job["with"].get("upload-test-run-reports", False), opted_in
                )
                permissions = {"id-token": "write", "contents": "read"}
                if opted_in:
                    permissions["actions"] = "read"
                    self.assertEqual(job["with"]["build-tool"], "cmake")
                    self.assertNotIn("editable", job["with"])
                self.assertEqual(job["permissions"], permissions)

    def test_no_other_job_gains_actions_read(self):
        selected = {("pull.yml", "unittest"), ("trunk.yml", "unittest-release")}
        for name in ("pull.yml", "trunk.yml"):
            for key, job in load_workflow(name)["jobs"].items():
                if (name, key) not in selected:
                    self.assertNotIn("actions", job.get("permissions", {}))

    def test_image_resolution_and_linux_entrypoint_are_preserved(self):
        jobs = load_workflow("_unittest.yml")["jobs"]
        self.assertEqual(jobs["linux"]["needs"], "docker-image")
        self.assertEqual(
            jobs["docker-image"]["uses"], "./.github/workflows/_docker-image.yml"
        )
        script = jobs["linux"]["with"]["script"]
        self.assertIn('.ci/scripts/unittest-linux.sh --build-tool "', script)
        self.assertIn(' --build-mode "${{ inputs.build-mode }}"', script)
        self.assertIn(' --editable "${{ inputs.editable }}"', script)

    def test_editable_and_buck_cannot_opt_in_even_with_a_true_input(self):
        expression = load_workflow("_unittest.yml")["jobs"]["linux"]["with"][
            "upload-test-run-reports"
        ][3:-2].strip()
        for enabled in (False, True):
            for tool in ("cmake", "buck2"):
                for editable in ("false", "true"):
                    with self.subTest(enabled=enabled, tool=tool, editable=editable):
                        evaluated = (
                            expression.replace(
                                "inputs.upload-test-run-reports", repr(enabled)
                            )
                            .replace("inputs.build-tool", repr(tool))
                            .replace("inputs.editable", repr(editable))
                            .replace("&&", "and")
                        )
                        self.assertEqual(
                            eval(evaluated, {"__builtins__": {}}, {}),
                            enabled and tool == "cmake" and editable == "false",
                        )
