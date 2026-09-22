import os
import tempfile
import unittest

import numpy as np
import torch
from executorch.backends.test.harness.tester import Tester


class GoldenArtifactTests(unittest.TestCase):
    def _dump(self, inputs, reference_output):
        artifact_dir = tempfile.mkdtemp()
        Tester._dump_golden_artifacts(artifact_dir, "m", inputs, reference_output)
        return artifact_dir

    def _read(self, artifact_dir, name):
        return np.fromfile(os.path.join(artifact_dir, name), dtype=np.float32)

    def test_channels_last_input_is_written_as_nhwc(self):
        x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
        channels_last = x.to(memory_format=torch.channels_last)

        artifact_dir = self._dump((channels_last,), channels_last)

        self.assertTrue(
            np.array_equal(
                self._read(artifact_dir, "m_input.bin"),
                channels_last.permute(0, 2, 3, 1).reshape(-1).numpy(),
            )
        )

    def test_contiguous_input_is_unchanged(self):
        x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)

        artifact_dir = self._dump((x,), x)

        self.assertTrue(
            np.array_equal(
                self._read(artifact_dir, "m_input.bin"), x.reshape(-1).numpy()
            )
        )

    def test_output_of_a_view_is_materialized(self):
        # reference_output is the eager result, so a model ending in permute
        # returns a view. The program materializes that contiguously, so the
        # golden output has to follow the values rather than the source layout.
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        permuted = x.permute(1, 0)

        artifact_dir = self._dump((x,), permuted)

        self.assertTrue(
            np.array_equal(
                self._read(artifact_dir, "m_expected_output.bin"),
                torch.tensor([0.0, 3.0, 1.0, 4.0, 2.0, 5.0]).numpy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
