# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
import warnings
from typing import Any, Callable, Optional

from executorch.exir._warnings import deprecated, experimental, ExperimentalWarning

#
# Classes
#


class UndecoratedClass:
    pass


@deprecated("DeprecatedClass message")
class DeprecatedClass:
    pass


@experimental("ExperimentalClass message")
class ExperimentalClass:
    pass


#
# Functions
#


def undecorated_function() -> None:
    pass


@deprecated("deprecated_function message")
def deprecated_function() -> None:
    pass


@experimental("experimental_function message")
def experimental_function() -> None:
    pass


#
# Methods
#


class TestClass:
    def undecorated_method(self) -> None:
        pass

    @deprecated("deprecated_method message")
    def deprecated_method(self) -> None:
        pass

    @experimental("experimental_method message")
    def experimental_method(self) -> None:
        pass


# NOTE: Variables and fields cannot be decorated.


class TestApiLifecycle(unittest.TestCase):

    def is_deprecated(
        self,
        callable: Callable[[], Any],  # pyre-ignore[2]: Any type
        message: Optional[str] = None,
    ) -> bool:
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Try to trigger a warning.
            callable()

            if not w:
                # No warnings were triggered.
                return False
            if not issubclass(w[-1].category, DeprecationWarning):
                # There was a warning, but it wasn't a DeprecationWarning.
                return False
            if issubclass(w[-1].category, ExperimentalWarning):
                # ExperimentalWarning is a subclass of DeprecationWarning.
                return False
            if message:
                return message in str(w[-1].message)
            return True

    def is_experimental(
        self,
        callable: Callable[[], Any],  # pyre-ignore[2]: Any type
        message: Optional[str] = None,
    ) -> bool:
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Try to trigger a warning.
            callable()

            if not w:
                # No warnings were triggered.
                return False
            if not issubclass(w[-1].category, ExperimentalWarning):
                # There was a warning, but it wasn't an ExperimentalWarning.
                return False
            if message:
                return message in str(w[-1].message)
            return True

    def test_undecorated_class(self) -> None:
        self.assertFalse(self.is_deprecated(UndecoratedClass))
        self.assertFalse(self.is_experimental(UndecoratedClass))

    def test_deprecated_class(self) -> None:
        self.assertTrue(self.is_deprecated(DeprecatedClass, "DeprecatedClass message"))
        self.assertFalse(self.is_experimental(DeprecatedClass))

    def test_experimental_class(self) -> None:
        self.assertFalse(self.is_deprecated(ExperimentalClass))
        self.assertTrue(
            self.is_experimental(ExperimentalClass, "ExperimentalClass message")
        )

    def test_undecorated_function(self) -> None:
        self.assertFalse(self.is_deprecated(undecorated_function))
        self.assertFalse(self.is_experimental(undecorated_function))

    def test_deprecated_function(self) -> None:
        self.assertTrue(
            self.is_deprecated(deprecated_function, "deprecated_function message")
        )
        self.assertFalse(self.is_experimental(deprecated_function))

    def test_experimental_function(self) -> None:
        self.assertFalse(self.is_deprecated(experimental_function))
        self.assertTrue(
            self.is_experimental(experimental_function, "experimental_function message")
        )

    def test_undecorated_method(self) -> None:
        tc = TestClass()
        self.assertFalse(self.is_deprecated(tc.undecorated_method))
        self.assertFalse(self.is_experimental(tc.undecorated_method))

    def test_deprecated_method(self) -> None:
        tc = TestClass()
        self.assertTrue(
            self.is_deprecated(tc.deprecated_method, "deprecated_method message")
        )
        self.assertFalse(self.is_experimental(tc.deprecated_method))

    def test_experimental_method(self) -> None:
        tc = TestClass()
        self.assertFalse(self.is_deprecated(tc.experimental_method))
        self.assertTrue(
            self.is_experimental(tc.experimental_method, "experimental_method message")
        )
