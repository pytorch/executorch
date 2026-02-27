import contextlib
import functools
import logging
import traceback
import typing as t
from contextvars import ContextVar
from dataclasses import dataclass, field

_LOGGER = logging.getLogger(__name__)

# A context variable to hold registered callbacks for all the observers for the
# current execution context. The callbacks list could have been a member
# variable on the observer instance, however, contextvars document advice
# against creating context variables not at module-global level.
# https://docs.python.org/3/library/contextvars.html#contextvars.ContextVar
_CALLBACKS: ContextVar[t.Dict["Observer", t.List[t.Callable]]] = ContextVar(
    "_CALLBACKS", default=None
)

TObserverCallback = t.TypeVar("TObserverCallback", bound=t.Callable[..., t.Any])

# Whether to rethrow the exception caught while calling observer callbacks.
# Default to False. True is only used during tests.
RETHROW_CALLBACK_EXCEPTION: bool = False


@dataclass(frozen=True)
class Observer(t.Generic[TObserverCallback]):
    """
    Usage:

    >>> some_observer: Observer = ...
    >>> with some_observer.add(callback_func):
    >>>     # do stuff, and when some_observer.observe() is called,
    >>>     # it will execute callback_func()
    >>>     ...

    """

    name: str = ""
    # Ensure each Observer instance is considered a distinct key when stored in
    # the `_CALLBACKS` dictionary.
    unique_id: object = field(default_factory=lambda: object())

    def add(self, callback: TObserverCallback) -> t.ContextManager:
        self._get_callbacks().append(callback)

        # Cannot decorate the outer `add` directly with `contextmanager`,
        # because if it were not used with a `with` statement, its body won't
        # be executed.
        @contextlib.contextmanager
        def _add():
            try:
                yield
            finally:
                try:
                    self._get_callbacks().remove(callback)
                except ValueError:
                    # Callback should be in the callbacks list. I'm just being
                    # extra cautious here. I don't want it to throw and affect
                    # business logic.
                    pass

        return _add()

    def observe(self, *args, **kwargs) -> None:
        for callback in self._get_callbacks():
            with _log_error(
                "Error calling observer callback", rethrow=RETHROW_CALLBACK_EXCEPTION
            ):
                callback(*args, **kwargs)

    def _get_callbacks(self) -> t.List[t.Callable]:
        """
        Gets the callbacks registered in current execution context. Any code
        that manipulates the returned list (add, remove, iterate) is
        concurrency safe.
        """
        callbacks_dict = _CALLBACKS.get()
        if callbacks_dict is None:
            callbacks_dict = {}
            _CALLBACKS.set(callbacks_dict)

        if self not in callbacks_dict:
            callbacks_dict[self] = []

        return callbacks_dict[self]


@dataclass(frozen=True)
class ObserveContext:
    """
    Passed to the registered callables that observes any function decorated by
    `observable`. See `observable` for detail.

    Attributes:
        callable: the observed callable object
        args: the args passed to the callable
        kwargs: the kwargs passed to the callable
        return_value: the return value returned by the callable, only available
            when observing the callable after its invocation (via
            `CallableObservers.post`)
    """

    callable: t.Callable
    args: t.List[t.Any]
    kwargs: t.Mapping[str, t.Any]
    return_value: t.Any = None


def observable():
    """
    A decorator to turn a function into observable

    Example:

    >>> @observable()
    >>> def func_to_observe(x, y) -> int:
    >>>     ...
    >>>
    >>> def log(ctx: ObserveContext):
    >>>     print(
    >>>         f"called {ctx.callable.__name__} with {ctx.args} {ctx.kwargs}"
    >>>     )
    >>>
    >>> # register:
    >>> with func_to_observe.observers.pre.add(log):
    >>>     func_to_observe(1, 2)
    >>>     # print out "called func_to_observe with (1,2)
    >>> # here it won't print
    """

    def decorator(observed_func: callable) -> ObservedCallable:
        wrapped_func = _make_observable(orig_func=observed_func)
        return functools.wraps(observed_func)(wrapped_func)

    return decorator


@dataclass(frozen=True)
class CallableObservers:
    pre: Observer[t.Callable[[ObserveContext], None]]
    post: Observer[t.Callable[[ObserveContext], None]]


class ObservedCallable:
    """
    Interface for an observed callable
    """

    observers: CallableObservers
    orig_func: callable

    def __call__(self, *args, **kwargs) -> t.Any:
        raise NotImplementedError()


def _make_observable(orig_func: t.Callable) -> ObservedCallable:
    """
    A wrapper for a callable which is to be observed.
    """

    observers = CallableObservers(
        pre=Observer(),
        post=Observer(),
    )

    @functools.wraps(orig_func)
    def observed_func(*args, **kwargs):
        observers.pre.observe(ObserveContext(orig_func, args, kwargs))
        return_value = None
        try:
            return_value = orig_func(*args, **kwargs)
            return return_value
        finally:
            observers.post.observe(
                ObserveContext(orig_func, args, kwargs, return_value)
            )

    observed_func.orig_func = orig_func
    observed_func.observers = observers

    return observed_func


@contextlib.contextmanager
def _log_error(msg: str, rethrow: bool = False) -> t.ContextManager:
    try:
        yield
    except Exception as e:
        _e = e  # noqa: F841
        _LOGGER.info(f"{msg} (This error is handled): {traceback.format_exc()}")
        if rethrow:
            raise
