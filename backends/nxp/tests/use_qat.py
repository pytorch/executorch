import pytest


@pytest.fixture
def use_qat(request):
    return request.param


def pytest_generate_tests(metafunc):
    if "use_qat" in metafunc.fixturenames:
        metafunc.parametrize("use_qat", [True, False], indirect=True)
