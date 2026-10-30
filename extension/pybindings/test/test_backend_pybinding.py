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

    def test_set_and_get_option(self) -> None:
        registry = Runtime.get().backend_registry
        before = registry.get_option("XnnpackBackend", {"weight_cache_enabled": False})
        try:
            registry.set_option(
                "XnnpackBackend",
                {"weight_cache_enabled": not before["weight_cache_enabled"]},
            )
            self.assertEqual(
                registry.get_option("XnnpackBackend", {"weight_cache_enabled": False}),
                {"weight_cache_enabled": not before["weight_cache_enabled"]},
            )
        finally:
            registry.set_option("XnnpackBackend", before)

    def test_set_option_errors(self) -> None:
        registry = Runtime.get().backend_registry
        # XNNPACK rejects a bool where it expects an int.
        with self.assertRaises(RuntimeError):
            registry.set_option("XnnpackBackend", {"workspace_sharing_mode": True})
        with self.assertRaises(TypeError):
            registry.set_option("XnnpackBackend", {"workspace_sharing_mode": 1.0})
        with self.assertRaises(RuntimeError):
            registry.set_option("NonExistBackend", {"key": 1})
        with self.assertRaises(RuntimeError):
            registry.get_option("NonExistBackend", {"key": 1})
