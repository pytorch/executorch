"""
A simple implementation of `select`, so that it's both available internally and in oss.
"""

load(":type_defs.bzl", "is_select")

def _apply_helper(function, inner):
    if not is_select(inner):
        return function(inner)
    return _apply(inner, function)

def _apply(obj, function):
    """
    If the object is a select, runs `select_map` with `function`.
    Otherwise, if the object is not a select, invokes `function` on `obj` directly.
    """
    if not is_select(obj):
        return function(obj)

    # @lint-ignore BUCKLINT
    return native.select_map(
        obj,
        native.partial(_apply_helper, function),
    )

selects = struct(
    apply = _apply,
)
