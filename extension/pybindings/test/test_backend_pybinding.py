import unittest

from executorch.runtime import Runtime


class TestBackendsPybinding(unittest.TestCase):
    def test_backend_name_list(
        self,
    ) -> None:

        runtime = Runtime.get()
        registered_backend_names = runtime.backend_registry.registered_backend_names
        self.assertGreaterEqual(len(registered_backend_names), 1)
        self.assertIn("XnnpackBackend", registered_backend_names)
