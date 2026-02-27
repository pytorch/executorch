from typing import Any, Callable, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

_DEBUG_ENABLED_FUNCS = []
_DEBUG_ENABLED_CLS = []


def fn_supports_debugger(func: F) -> F:
    _DEBUG_ENABLED_FUNCS.append(func)
    return func


def cls_supports_debugger(cls: Type[T]) -> Type[T]:
    _DEBUG_ENABLED_CLS.append(cls)
    return cls
