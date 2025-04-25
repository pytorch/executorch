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

    def test_backend_is_available(
        self,
    ) -> None:
        # XnnpackBackend is available
        runtime = Runtime.get()
        self.assertTrue(
            runtime.backend_registry.is_available(backend_name="XnnpackBackend")
        )
        # NonExistBackend doesn't exist and not available
        self.assertFalse(
            runtime.backend_registry.is_available(backend_name="NonExistBackend")
        )
