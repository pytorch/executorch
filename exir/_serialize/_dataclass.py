# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
import json
from dataclasses import fields, is_dataclass
from typing import Any, Dict, get_args, get_origin, get_type_hints, Union


class _DataclassEncoder(json.JSONEncoder):
    # pyre-ignore
    def default(self, o: Any) -> Any:
        if is_dataclass(o):
            props = {}
            for field in fields(o):
                props[field.name] = getattr(o, field.name)
                origin = get_origin(get_type_hints(type(o))[field.name])
                if isinstance(field.type, str) and origin == Union:
                    props[f"{field.name}_type"] = type(getattr(o, field.name)).__name__
            return props

        if isinstance(o, bytes):
            return list(o)

        return super().default(o)


# Dataclass Decoder
# pyre-ignore
def _is_optional(T: Any) -> bool:
    return (
        get_origin(T) is Union
        and len(get_args(T)) > 0
        and isinstance(None, get_args(T)[-1])
    )


# pyre-ignore
def _is_strict_union(T: Any, cls: Any, key: str) -> bool:
    return isinstance(T, str) and get_origin(get_type_hints(cls)[key]) is Union


# pyre-ignore
def _get_class_from_union(json_dict: Dict[str, Any], key: str, cls: Any) -> Any:
    """Search through all possible types in the Union and select the type we
    want to unpack (note in the serialization of a PyObject to JSON,
    the type we want to unpack is keyed by f"{field.name}_type").
    """
    _type = json_dict[key + "_type"]
    res = [x for x in get_args(get_type_hints(cls)[key]) if x.__name__ == _type]
    return res[0]


# pyre-ignore
def _json_to_dataclass(json_dict: Dict[str, Any], cls: Any = None) -> Any:
    """Initializes a dataclass given a dictionary loaded from a json,
    `json_dict`, and the expected class, `cls`, by iterating through the fields
    of the class and retrieving the data for each. If there is a field that is
    missing in the data, and that field is not the Optional type,
    `_json_to_dataclass` raises a TypeError.

    Args:
        `json_dict` : Dictionary formatted to represent a class, where fields are keys in the dictionary,
        and values are values with the required type (as outlined in the dataclass definition). If a field is
        specified to be another dataclass, the value will be another dictionary. See example below:

        SAMPLE JSON:
        {field1 : v1, inner_class : {field2_1: v2_1, field2_2: v2_1}}.

        `cls` : The class that we should be unpacking from the given dictionary
                (gives us an idea of what fields and values will be present in `json_dict`)

        SAMPLE CLASSES for Above JSON:
        @dataclass
        class AnotherDataClass
            field2_1: int
            field2_2: str

        @dataclass
        class Example
            field1 : str
            inner_class: AnotherDataClass

    Returns: An initialized PyObject of class: `cls`, given the data from `json_dict`.
    """
    if not is_dataclass(cls) or is_dataclass(json_dict):
        return json_dict

    # initialize dataclass by iterating through all required fields
    cls_flds = fields(cls)
    data = {}
    for field in cls_flds:
        key = field.name
        T = field.type

        if _is_optional(T):
            T = get_args(T)[0]
            value = json_dict.get(key, None)
        elif _is_strict_union(T, cls, key):
            # If the field is a Union type, we determine exactly what type we
            # are trying to initialize by calling `_get_class_from_union`, and
            # then make a recursive call construct this new class
            _cls = _get_class_from_union(json_dict, key, cls)
            data[key] = _json_to_dataclass(json_dict[key], _cls)
            continue
        else:
            try:
                value = json_dict[key]
            except KeyError:
                raise TypeError(
                    f"Invalid Buffer. Received no value for field: {key}, but {key} : {T} is not an Optional type."
                )

        if value is None:
            data[key] = None
            continue

        if is_dataclass(T):
            data[key] = _json_to_dataclass(value, T)
            continue

        if get_origin(T) is list:
            T = get_args(T)[0]
            data[key] = [_json_to_dataclass(e, T) for e in value]
            continue

        # If T is a Union, then check which type in the Union it is and initialize.
        # eg. Double type in schema.py
        if get_origin(T) is Union:
            res = [x for x in get_args(get_type_hints(cls)[key]) if x == type(value)]
            data[key] = res[0](value)
            continue

        # If T is an enum then lookup the value in the enum otherwise try to
        # cast value to whatever type is required
        if isinstance(T, enum.EnumMeta):
            data[key] = T[value]
        else:
            data[key] = T(value)
    return cls(**data)
