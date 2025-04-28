"""Provides macros for querying type information."""

_SELECT_TYPE = type(select({"DEFAULT": []}))

def is_select(thing):
    return type(thing) == _SELECT_TYPE

def is_unicode(arg):
    """Checks if provided instance has a unicode type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for unicode instances, False otherwise. rtype: bool
    """
    return hasattr(arg, "encode")

_STRING_TYPE = type("")

def is_string(arg):
    """Checks if provided instance has a string type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for string instances, False otherwise. rtype: bool
    """
    return type(arg) == _STRING_TYPE

_LIST_TYPE = type([])

def is_list(arg):
    """Checks if provided instance has a list type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for list instances, False otherwise. rtype: bool
    """
    return type(arg) == _LIST_TYPE

_DICT_TYPE = type({})

def is_dict(arg):
    """Checks if provided instance has a dict type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for dict instances, False otherwise. rtype: bool
    """
    return type(arg) == _DICT_TYPE

_TUPLE_TYPE = type(())

def is_tuple(arg):
    """Checks if provided instance has a tuple type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for tuple instances, False otherwise. rtype: bool
    """
    return type(arg) == _TUPLE_TYPE

type_utils = struct(
    is_string = is_string,
    is_unicode = is_unicode,
    is_list = is_list,
    is_dict = is_dict,
    is_tuple = is_tuple,
    is_select = is_select,
)
