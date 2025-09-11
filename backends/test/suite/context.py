# Test run context management. This is used to determine the test context for reporting
# purposes.
class TestContext:
    subtest_index: int

    def __init__(
        self, test_name: str, test_base_name: str, flow_name: str, params: dict | None
    ):
        self.test_name = test_name
        self.test_base_name = test_base_name
        self.flow_name = flow_name
        self.params = params
        self.subtest_index = 0

    def __enter__(self):
        global _active_test_context
        import sys

        if _active_test_context is not None:
            print(f"Active context: {_active_test_context.test_name}", file=sys.stderr)
        assert _active_test_context is None
        _active_test_context = self

    def __exit__(self, exc_type, exc_value, traceback):
        global _active_test_context
        _active_test_context = None


_active_test_context: TestContext | None = None


def get_active_test_context() -> TestContext | None:
    global _active_test_context
    return _active_test_context
