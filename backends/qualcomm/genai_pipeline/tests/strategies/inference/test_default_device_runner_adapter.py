# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)

from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.inference.runner_io import (
    ExecuteOutput,
    ExecuteRequest,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.default_device_runner_adapter import (
    DefaultDeviceRunnerAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.tests.test_utils import (
    TEST_PTE_PATH,
    TEST_SOC_CHIPSET,
)

_DEVICE = "executorch.backends.qualcomm.export_utils.Device"


def _make_input_config():
    """Create an InferenceInputConfig holding one decoder PTE."""
    return InferenceInputConfig(
        soc_model=TEST_SOC_CHIPSET,
        artifact_dir=Path("/tmp/artifacts"),
        pte_paths={ARTIFACT_TEXT_DECODER: TEST_PTE_PATH},
    )


class TestDefaultDeviceRunnerAdapter(unittest.TestCase):
    def setUp(self):
        patcher = patch(_DEVICE)
        self.mock_device_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.device = self.mock_device_cls.return_value
        self.qnn_config = MagicMock(name="qnn_config")
        self.adapter = DefaultDeviceRunnerAdapter(self.qnn_config, workspace="/ws")
        self.request = ExecuteRequest(inputs=((1,), (2,)), method_index=1, iteration=3)

    def _push_artifacts(self, **kwargs):
        """Deploy the adapter with the default input configuration."""
        self.adapter.push_artifacts(ControlArgs(), _make_input_config(), **kwargs)

    def test_default_workspace_is_per_user(self):
        """The default workspace is a per-user folder."""
        DefaultDeviceRunnerAdapter(self.qnn_config).push_artifacts(
            ControlArgs(), _make_input_config()
        )
        self.assertEqual(
            self.mock_device_cls.call_args.kwargs["workspace"],
            f"/data/local/tmp/{getpass.getuser()}/genai_pipeline",
        )

    def test_push_artifacts_pushes_every_pte(self):
        """Every PTE is handed to Device, with the up-front inputs and files."""
        self._push_artifacts(input_data=[(1,)], extra_files=["/host/f"])
        self.mock_device_cls.assert_called_once_with(
            qnn_config=self.qnn_config,
            pte_path=[str(TEST_PTE_PATH)],
            workspace="/ws",
        )
        self.device.push.assert_called_once_with(inputs=[(1,)], files=["/host/f"])

    def test_get_metadata_is_none(self):
        """The generic executor runner has no decoder metadata."""
        self.assertIsNone(self.adapter.get_metadata())

    def test_push_inputs_pushes_samples_without_reinitializing(self):
        """A request's samples are pushed as the input list without clearing the workspace."""
        self._push_artifacts()
        self.adapter.push_inputs(self.request)
        self.device.push.assert_called_with(inputs=[(1,), (2,)], init_env=False)

    def test_execute_runs_requested_method(self):
        """The request's method index and iteration count reach Device."""
        self._push_artifacts()
        self.adapter.execute(self.request)
        self.device.execute.assert_called_once_with(method_index=1, iteration=3)

    def test_pull_results_collects_pulled_files(self):
        """Files pulled into a subdirectory are flattened into the ExecuteOutput."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"

            def fake_pull(host_output_path):
                pulled = Path(host_output_path) / "outputs"
                pulled.mkdir(parents=True)
                (pulled / "output_0_0.raw").write_bytes(b"0")

            self.device.pull.side_effect = fake_pull
            self._push_artifacts()
            result = self.adapter.pull_results(self.request, output_dir)
        self.assertIsInstance(result, ExecuteOutput)
        self.assertEqual(result.files, [output_dir / "outputs" / "output_0_0.raw"])

    def test_methods_raise_before_push_artifacts(self):
        """Lifecycle methods raise RuntimeError before push_artifacts."""
        for call in (
            lambda: self.adapter.push_inputs(self.request),
            lambda: self.adapter.execute(self.request),
            lambda: self.adapter.pull_results(self.request, Path("/tmp/out")),
        ):
            with self.assertRaisesRegex(RuntimeError, "push_artifacts"):
                call()


if __name__ == "__main__":
    unittest.main()
