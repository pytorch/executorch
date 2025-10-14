#
# Copyright 2023 Martin Pavella
# Copyright 2023-2025 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    logger

Module implements functions for logging, error messages and custom assertions.
"""

import sys
from collections import defaultdict
from enum import Enum
from typing import NoReturn, Optional


class Style:
    """Strings used to set a color and other styles to the output printed to console.

    example usage:
        logger.w(f'{logger.Style.orange + logger.Style.bold}Some warning. {logger.Style.end}Additional info.')

    """

    red = "\033[91m"
    green = "\033[92m"
    orange = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    cyan = "\033[96m"

    bold = "\033[1m"
    underline = "\033[4m"

    end = "\033[0m"


class MessageImportance(Enum):
    """Importance levels of messages to print."""

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


MIN_OUTPUT_IMPORTANCE = MessageImportance.WARNING


class Message:
    """Custom messages, that are printed to console from different locations in the code."""

    ALLOW_SELECT_OPS = (
        "If you want to convert the model using the SELECT_TF_OPS, run the conversion again with "
        f"the flag {Style.bold + Style.cyan}--allow-select-ops{Style.end}."
    )

    GUARANTEE_NON_NEGATIVE_INDICES = (
        f"{Style.green}If you know that the indices are always non-negative, you can run"
        f" the converter with the flag {Style.bold + Style.cyan}--non-negative-indices"
        f"{Style.end}."
    )

    CAST_INT64_TO_INT32 = (
        f"Use option {Style.bold + Style.cyan}--cast-int64-to-int32{Style.end} to disable this "
        "check and re-cast input/output to INT32."
    )

    IGNORE_OPSET_VERSION = (
        "If you want to try and convert the model anyway, run the conversion again with the flag "
        f"{Style.bold + Style.cyan}--ignore-opset-version{Style.end}. Keep in mind that the output"
        " TFLite model may potentially be invalid."
    )


class Code(Enum):
    """Error codes"""

    INTERNAL_ERROR = 1
    GENERATED_MODEL_INVALID = 2
    INVALID_OPTIMIZATION = 3
    PREPROCESSING_ERROR = 4

    UNSUPPORTED_OPERATOR = 21
    # Code 22 was removed.
    UNSUPPORTED_OPERATOR_ATTRIBUTES = 23
    NOT_IMPLEMENTED = 24

    INVALID_TYPE = 31
    INVALID_TENSOR_SHAPE = 32
    # Code 33 was removed.
    INVALID_OPERATOR_ATTRIBUTE = 34
    INVALID_INPUT_MODEL = 35

    CONVERSION_IMPOSSIBLE = 41
    # Code 42 was removed.
    IO_PRESERVATION_ERROR = 43

    INVALID_INPUT = 51

    UNSUPPORTED_NODE = 61


class Error(Exception):

    def __init__(self, err_code: Code, msg, exception: Optional[Exception] = None):
        self.error_code = err_code
        self.msg = msg
        self.exception = exception

    def __str__(self):
        output = f"[{self.error_code}] - {self.msg}"
        if self.exception is not None:
            output += f" - (Parent exception: {self.exception})"

        return output


class LoggingContext:
    """
    Context that represents part of an application to which current logs belong to. Contexts are meant
    to be nested from most general (global) to most specific (node context etc.). Use context manager
    'logger.loggingContext()' to enable specific context.
    """

    def __init__(self, context_name):
        self.context_name = context_name

    def __str__(self) -> str:
        return self.context_name

    def __repr__(self) -> str:
        return self.context_name


class BasicLoggingContext(LoggingContext):
    """
    Basic logging contexts specified by its name.
    """

    GLOBAL = LoggingContext("global")
    OPERATOR_CONVERSION = LoggingContext("operator_conversion")
    TFLITE_GENERATOR = LoggingContext("tflite_generator")
    QDQ_QUANTIZER = LoggingContext("qdq_quantizer")


class NodeLoggingContext(LoggingContext):
    """
    ExecuTorch node specific context. Logs reported within this context are related to node with index 'node_id'.
    """

    def __init__(self, node_id):
        self.node_id = node_id
        super().__init__(f"node_{node_id}")


class ConversionLog:
    """
    Record logs sent within some logging context. Log might belong to multiple contexts. Single log
    event are present with: message, logging context hierarchy, importance (logger.MessageImportance) and
    optional error code (logger.Code). Logs added outside any context are ignored.
    """

    _current_logging_context = []
    _log = defaultdict(list)
    _log_count = 0

    def append_context(self, loggingContext: LoggingContext):
        if len(self._current_logging_context) == 0:
            self._log = defaultdict(list)
            self._log_count = 0

        self._current_logging_context.append(loggingContext.context_name)

    def pop_last_context(self):
        self._current_logging_context.pop()

    def reset(self):
        self._log = defaultdict(list)
        self._current_logging_context = []
        self._log_count = 0

    def add_log(
        self,
        importance: MessageImportance,
        message: str,
        error_code: Code | None = None,
    ):
        data = {
            "message": message,
            "logging_context_hierarchy": list(self._current_logging_context),
            "importance": importance.value,
            "message_id": self._log_count,
        }

        if error_code is not None:
            data["error_code"] = error_code

        if len(self._current_logging_context) != 0:
            self._log[self._current_logging_context[-1]].append(data)
            self._log_count += 1

    def get_logs(self) -> dict:
        return self._log

    def _get_node_error(self, node_id: int, dict_item: str) -> Code | str | None:
        """
        Return first error log item that belong to node with id 'node_id'. If no error is present
        None is returned instead.

        :param node_id: ExecuTorch node id.
        :param dict_item: Dictionary item to return from `log`
        :return: Error code or None if there's no error related to node.
        """

        node_logs = self._log[f"node_{node_id}"]
        for log in node_logs:
            if log["importance"] == MessageImportance.ERROR.value:
                return log[dict_item]

        return None

    def get_node_error_code(self, node_id: int) -> Code | None:
        """
        Return first error code that belong to node with id 'node_id'. If no error is present
        None is returned instead.

        :param node_id: ExecuTorch node id.
        :return: Error code or None if there's no error related to node.
        """

        return self._get_node_error(node_id, "error_code")

    def get_node_error_message(self, node_id: int) -> str | None:
        """
        Return first error message that belong to node with id 'node_id'. If no error is present
        None is returned instead.

        :param node_id: ExecuTorch node id
        :return: Error message or None if there is no error related to node.
        """

        return self._get_node_error(node_id, "message")


conversion_log = ConversionLog()


class loggingContext:
    """
    Context manager used to nest logging contexts. Usage:

    with loggingContext(BasicLoggingContext.GLOBAL):
        with loggingContext(BasicLoggingContext.OPERATOR_CONVERSION):
            logger.i("My log") # this log is automatically assigned to both parent contexts

    """

    def __init__(self, logging_context: LoggingContext):
        self.logging_context = logging_context

    def __enter__(self):
        conversion_log.append_context(self.logging_context)

    def __exit__(self, _, __, ___):
        conversion_log.pop_last_context()


def d(msg: str):
    """Log internal debug message with given parameters."""

    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.DEBUG.value:
        return

    print("DEBUG: ", msg)
    conversion_log.add_log(MessageImportance.DEBUG, msg)


def i(msg: str):
    """Log info message with given parameters."""

    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.INFO.value:
        return

    print("INFO: ", msg)
    conversion_log.add_log(MessageImportance.INFO, msg)


def w(msg: str):
    """Log warning message with given parameters."""

    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.WARNING.value:
        return

    print("WARNING: ", msg)
    conversion_log.add_log(MessageImportance.WARNING, msg)


def e(err_code: Code, msg: str, exception: Optional[Exception] = None) -> NoReturn:
    """Print and raise exception with error message composed of provided error code, messages and optional exception.
    :param err_code: Error code.
    :param msg: Error message.
    :param exception: (Optional) Exception object to print before the program exits.
    """

    error = Error(err_code, msg, exception)
    conversion_log.add_log(MessageImportance.ERROR, str(error), error_code=err_code)
    print("ERROR: ", str(error), file=sys.stderr)

    raise error


def expect_type(obj, expected_type, msg: str = ""):
    if type(obj) is not expected_type:
        w(
            msg
            + f":Object '{obj}' is of type '{type(obj)}' where '{expected_type}' was expected!"
        )


def require_type(obj, required_type, msg: str = ""):
    if type(obj) is not required_type:
        e(
            Code.INVALID_TYPE,
            msg
            + f":Object '{obj}' is of type '{type(obj)}' where '{required_type}' was required!",
        )


def internal_assert(truth_value: bool, msg: str = ""):
    """Assert that the 'truth_value' is True. If not, raise a logger INTERNAL_ERROR with message 'msg'.

    :param truth_value: Boolean to check.
    :param msg: Message to raise the Error with.
    """

    if not truth_value:
        e(Code.INTERNAL_ERROR, msg)
