import subprocess
import sys
import unittest


class TestNncfImport(unittest.TestCase):
    def test_partitioner_imports_without_nncf(self):
        # nncf is only needed to quantize, so exporting must work without it.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['nncf'] = None; "
                "from executorch.backends.openvino.partitioner import OpenvinoPartitioner",
            ],
            check=True,
        )
