#
# Copyright 2023 Martin Pavella
# Copyright 2024 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    meta

Implementations of classes that all classes in /src/tflite_generator/ inherit from.
"""
import logging
from typing import Callable, Iterator, List, Union

import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as bOp
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as bOpt

import flatbuffers as fb

logger = logging.getLogger(__name__)

""" This file contains parent classes for simple classes used in the '/model' directory. """


class TFLiteObject:
    """Parent class for all tflite objects. That is all objects in the 'tflite_generator' directory."""

    """ Generates tflite representation for this object. MUST be overridden! """

    def gen_tflite(self, builder: fb.Builder) -> int:
        logger.warning("TFLiteObject: genTFLite() is not defined!")
        return 0


class TFLiteVector(TFLiteObject):
    """Represents a TFLite vector of TFLiteObjects. Provides interface for storing data
    and generating output TFLite code."""

    vector: List[Union[TFLiteObject, int, float, bool]]

    """ Indicates if an empty vector should be generated if 'vector' attribute is
    empty, or to not generate anything in that case. """
    gen_empty: bool = True

    """ TFLite 'Start...Vector' function for the exact vector. Takes 2 arguments, 
    'flatbuffers.Builder' and number of vector elements """
    start_function: Callable[[fb.Builder, int], None]

    """ TFLite 'Prepend...' function for the exact vector item type. Takes 'flatbuffers.Builder' 
    as argument """
    prepend_function: Callable[[fb.Builder], Callable[[int], None]]

    def __init__(
        self,
        vector: List[Union[TFLiteObject, int, float, bool]],
        start_function: Callable[[fb.Builder, int], None],
        prepend_function: Callable[
            [fb.Builder], Callable[[int], None]
        ] = lambda builder: builder.PrependUOffsetTRelative,
        gen_empty: bool = True,
    ) -> None:
        if vector is None:
            vector = []
        self.vector = vector
        self.start_function = start_function
        self.prepend_function = prepend_function
        self.gen_empty = gen_empty

    def append(self, item):
        self.vector.append(item)

    def insert(self, index: int, item):
        self.vector.insert(index, item)

    def index(self, item) -> int:
        return self.vector.index(item)

    def remove(self, item):
        self.vector.remove(item)

    def get(self, index: int):
        return self.vector[index]

    def get_last(self):
        if len(self.vector) > 0:
            return self.vector[-1]
        return None

    def len(self):
        return self.vector.__len__()

    def __str__(self):
        return self.vector.__str__()

    def __iter__(self) -> Iterator:
        return self.vector.__iter__()

    def __getitem__(self, index):
        return self.vector[index]

    def gen_tflite(self, builder: fb.Builder):
        """Generates TFLite code for the vector"""

        if (not self.gen_empty) and (len(self.vector) == 0):
            # Nothing to generate
            return

        # IMPORTANT! tflite MUST be generated for list items in REVERSE ORDER!
        # Otherwise, the order will be wrong.
        tfl_vector = [item.gen_tflite(builder) for item in reversed(self.vector)]

        self.start_function(builder, len(self.vector))

        for tfl_item in tfl_vector:
            self.prepend_function(builder)(tfl_item)

        return builder.EndVector()


class TFLiteAtomicVector(TFLiteVector):
    def __init__(
        self,
        vector: List[Union[int, float, bool]],
        start_function: Callable[[fb.Builder, int], None],
        prepend_function: Callable[[fb.Builder], Callable[[int], None]],
        gen_empty: bool = True,
    ) -> None:
        super().__init__(vector, start_function, prepend_function, gen_empty)

    def __eq__(self, other):
        return self.vector == other.vector

    def gen_tflite(self, builder: fb.Builder):
        """Generates TFLite code for the vector"""

        if (not self.gen_empty) and (len(self.vector) == 0):
            # Nothing to generate
            return

        self.start_function(builder, len(self.vector))

        # IMPORTANT! tflite MUST be generated for list items in REVERSE ORDER!
        # Otherwise, the order will be wrong.
        for val in reversed(self.vector):
            self.prepend_function(builder)(val)

        return builder.EndVector()


class FloatVector(TFLiteAtomicVector):
    """Class represents a TFLite vector of float values. Provides interface for storing data
    and generating output TFLite code."""

    def __init__(
        self,
        float_list: List[float],
        start_function: Callable[[fb.Builder, int], None],
        prepend_function: Callable[
            [fb.Builder], Callable[[int], None]
        ] = lambda builder: builder.PrependFloat32,
        gen_empty: bool = True,
    ) -> None:
        super().__init__(float_list, start_function, prepend_function, gen_empty)


class IntVector(TFLiteAtomicVector):
    """Class represents a TFLite vector of integer values. Provides interface for storing data
    and generating output TFLite code."""

    vector: List[int]

    def __init__(
        self,
        int_list: List[int],
        start_function: Callable[[fb.Builder, int], None],
        prepend_function: Callable[
            [fb.Builder], Callable[[int], None]
        ] = lambda builder: builder.PrependInt32,
        gen_empty: bool = True,
    ) -> None:
        super().__init__(int_list, start_function, prepend_function, gen_empty)


class BoolVector(TFLiteAtomicVector):
    """Class represents a TFLite vector of boolean values. Provides interface for storing data
    and generating output TFLite code."""

    vector: List[bool]

    def __init__(
        self,
        bool_list: List[bool],
        start_function: Callable[[fb.Builder, int], None],
        prepend_function: Callable[
            [fb.Builder], Callable[[int], None]
        ] = lambda builder: builder.PrependBool,
        gen_empty: bool = True,
    ) -> None:
        super().__init__(bool_list, start_function, prepend_function, gen_empty)


class BuiltinOptions(TFLiteObject):
    """Class represents 'BuiltinOptions' for an Operator. Used in 'model/Operators.py'.
    Provides interface for work with any BuiltinOptions table.
    This class alone does NOT generate any TFLite.
    Subclasses do NOT generate TFLite for the 'builtinOptionsType', only for the exact options.
    'builtinOptionsType' is merely stored here for convenience and an 'Operator' object
    generates its TFLite representation (as it is the child of the 'operator' table in 'operators').
    """

    """ The type of parameters of this operator. """
    builtin_options_type: bOpt.BuiltinOptions

    """ The type of this operator. """
    operator_type: bOp.BuiltinOperator

    def __init__(
        self,
        builtin_options_type: bOpt.BuiltinOptions,
        operator_type: bOp.BuiltinOperator,
    ) -> None:
        if builtin_options_type is None:
            logger.d(
                "TFLITE: Operator inheriting from 'BuiltinOptions'. MUST specify the 'builtinOptionsType'!"
            )
        if operator_type is None:
            logger.d(
                "TFLITE: Operator inheriting from 'BuiltinOptions'. MUST specify the 'operatorType'!"
            )
        self.builtin_options_type = builtin_options_type
        self.operator_type = operator_type

    """ Function has to be overwritten """

    def gen_tflite(self, builder: fb.Builder):
        logger.w(
            f"BuiltinOperator '{self.builtin_options_type}':genTFLite() is not defined!"
        )


class CustomOptions(bytearray):
    """Class represents a `custom_options` object in the TFLite model, i.e. a bytearray form of the parameters of a
     `custom` TFLite operator.

    Currently, this is being used for `Flex Delegate` operators / `SELECT_TF_OPS`.
    """

    operator_type = bOp.BuiltinOperator.CUSTOM
    custom_code: str

    def __init__(self, custom_code: str, data: bytearray):
        super().__init__()
        self.custom_code = custom_code
        self[:] = data
