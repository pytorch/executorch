import json
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import coremltools as ct
import torch
from coremltools.converters.mil.frontend.torch.utils import TORCH_DTYPE_TO_MIL_DTYPE

_IGNORE_RANGE_CONSTRAINTS: bool = False


@dataclass(frozen=True, slots=True)
class _SymInt:
    key_name: str
    low: Optional[int]
    high: Optional[int]

    @classmethod
    def from_symint_and_range_constraints(cls, s: torch.SymInt, range_constraints=None):
        # Canonicalize: "Sym(s0)" -> "s0", or leave "s0" as is
        def _symkey(sym: torch.SymInt) -> str:
            s = str(sym)
            return s[4:-1] if s.startswith("Sym(") and s.endswith(")") else s

        # Convert symint to int.  Infinity is converted to None
        def _as_int_or_none(b):
            if b is None:
                return None
            s = str(b)
            if s in {"int_oo", "-int_oo", "oo", "-oo", "Infinity", "-Infinity"}:
                return None
            return int(s)

        # Get low/high from range_constraints if provided
        low, high = None, None
        if range_constraints is not None:
            for k, v in range_constraints.items():
                if _symkey(k) == _symkey(s):
                    low = _as_int_or_none(getattr(v, "lower", getattr(v, "min", None)))
                    high = _as_int_or_none(getattr(v, "upper", getattr(v, "max", None)))
        return _SymInt(_symkey(s), low, high)


@dataclass(frozen=True, slots=True)
class _SymbolicShape:
    shape: Tuple[int | _SymInt]

    @classmethod
    def from_shape_and_range_constraints(cls, shape, range_constraints=None):
        out_shape = []
        for s in shape:
            if isinstance(s, int):
                assert s >= 0
                out_shape.append(s)
            elif isinstance(s, torch.SymInt):
                out_shape.append(
                    _SymInt.from_symint_and_range_constraints(s, range_constraints)
                )
            else:
                raise ValueError(f"Unexpected type found in shape: {type(s)}")
        out_shape = tuple(out_shape)
        return _SymbolicShape(out_shape)

    def is_static_shape(self):
        for s in self.shape:
            if isinstance(s, _SymInt):
                return False
        return True

    def __len__(self):
        return len(self.shape)

    def __getitem__(self, key):
        return self.shape[key]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        assert len(d) == 1 and "shape" in d
        shape = []
        for s in d["shape"]:
            if isinstance(s, int):
                shape.append(s)
            elif isinstance(s, dict):
                assert len(s) == 3 and "key_name" in s and "low" in s and "high" in s
                shape.append(_SymInt(**s))
            else:
                raise ValueError(f"Unexpected type found in shape: {type(s)}")
        shape = tuple(shape)
        return _SymbolicShape(shape)


def _iterate_over_fake_user_inputs(ep):
    user_inputs = ep.graph_signature.user_inputs
    for node in ep.graph.nodes:
        if node.op == "placeholder" and node.name in user_inputs:
            yield (node.name, node.meta["val"])


def _create_enumeration_map(ep, enumerated_shapes, *, ignore_range_constraints=False):
    # Each input should have the same number of enumerations
    assert len(enumerated_shapes) > 0, "No enumerated shapes provided"
    num_enumerations = None
    for name, eshapes in enumerated_shapes.items():
        if num_enumerations is None:
            num_enumerations = len(eshapes)
        else:
            assert (
                len(eshapes) > 1
            ), f"Input {name} only has {len(eshapes)} enumerated shapes provided.  You should not specify enumerated shapes for inputs with only 1 input."
            assert (
                len(eshapes) == num_enumerations
            ), f"Input {name} has {len(eshapes)} enumerated shape provided, but other inputs have {num_enumerations} enumerated shapes"

    symbolic_shape_to_enumerations = {}
    for name, fake_input in _iterate_over_fake_user_inputs(ep):
        shape = fake_input.shape
        serialized_shape = _SymbolicShape.from_shape_and_range_constraints(
            shape, ep.range_constraints if not ignore_range_constraints else None
        )
        if serialized_shape.is_static_shape():
            continue
        # Shape is dynamic
        if name not in enumerated_shapes:
            raise ValueError(
                f"The input {name} has a symbolic shape, but you did not provide an enumeration for it"
            )
        # Validate
        for eshape in enumerated_shapes[name]:
            assert len(serialized_shape) == len(
                eshape
            ), f"In {name}, the rank of the enumeration is {len(eshape)}, but the symbolic shape has rank {len(serialized_shape)}"
            for i in range(len(eshape)):
                assert isinstance(
                    eshape[i], int
                ), f"Enumerated shapes must be ints, but got {type(eshape[i])}."
                assert eshape[i] >= 1, "Each enumerated shape dimension must be >= 1"
                if isinstance(serialized_shape[i], int):
                    assert (
                        serialized_shape[i] == eshape[i]
                    ), f"In {name}, the shape enumeration {eshape} does not match {shape} on the non-symbolic value at index {i}"
                else:
                    # Check eshape is within bound
                    if serialized_shape[i].low is not None:
                        # We add special case for when the low bound is 2.  This is because Torch does not usually allow 1 as a lower bound
                        assert (eshape[i] >= serialized_shape[i].low) or (
                            eshape[i] == 1 and serialized_shape[i].low == 2
                        ), f"In {name}, the shape enumeration {eshape} violates the lower range-constraint on the symbolic shape {shape} at index {i}"
                    if serialized_shape[i].high is not None:
                        assert (
                            eshape[i] <= serialized_shape[i].high
                        ), f"In {name}, the shape enumeration {eshape} violates the upper range-constraint on the symbolic shape {shape} at index {i}"
        if serialized_shape in symbolic_shape_to_enumerations:
            enumerations, names = symbolic_shape_to_enumerations[serialized_shape]
            assert (
                enumerations == enumerated_shapes[name]
            ), f"The symbolic shape {shape}, has multiple enumerations defined.  A new enumeration is defined for input {name}, but the existing inputs {names} have a different one defined.  If these inputs have different enumerations, they should be exported with different symbolic shapes."
            names.append(name)
            symbolic_shape_to_enumerations[serialized_shape] = (enumerations, names)
        else:
            symbolic_shape_to_enumerations[serialized_shape] = (
                enumerated_shapes[name],
                [name],
            )
    return symbolic_shape_to_enumerations


class _SymbolicShapeToEnumeratedShapeMap:
    def __init__(self, emap):
        self.emap = emap

    def to_json(self):
        json_list = []
        for k in self.emap:
            json_list.append((k.to_dict(), self.emap[k]))
        return json.dumps(json_list)

    @classmethod
    def from_json(cls, s):
        emap = {}
        json_list = json.loads(s)
        for k, v in json_list:
            k = _SymbolicShape.from_dict(k)
            emap[k] = tuple(v)
        return cls(emap)

    @classmethod
    def from_exported_program(
        cls,
        ep,
        enumerated_shapes,
        *,
        ignore_range_constraints=_IGNORE_RANGE_CONSTRAINTS,
    ):
        emap = _create_enumeration_map(
            ep, enumerated_shapes, ignore_range_constraints=ignore_range_constraints
        )
        return cls(emap)

    def __getitem__(self, key: _SymbolicShape):
        return self.emap[key][0]

    def __contains__(self, key):
        return key in self.emap

    def __repr__(self):
        return f"_SymbolicShapeToEnumeratedShapeMap(emap={self.emap})"


def _get_ct_inputs(ep, emap: _SymbolicShapeToEnumeratedShapeMap):
    ct_inputs = []
    for name, fake_input in _iterate_over_fake_user_inputs(ep):

        # CoreML can do funny conversions in ct.convert (e.g., int64 -> int32, int16 -> int32), so here
        # we restrict users to use dtypes we know are supported
        _ENUMERATED_SHAPE_INPUT_DTYPES = [torch.float16, torch.float32, torch.int32]
        for dtype in _ENUMERATED_SHAPE_INPUT_DTYPES:
            assert dtype in TORCH_DTYPE_TO_MIL_DTYPE
        assert (
            fake_input.dtype in _ENUMERATED_SHAPE_INPUT_DTYPES
        ), f"When using enumerated shapes, all inputs must have one of the following dtyeps {_ENUMERATED_SHAPE_INPUT_DTYPES}, but {name} has dtype {fake_input.dtype}"

        ct_dtype = TORCH_DTYPE_TO_MIL_DTYPE[fake_input.dtype]
        shape = fake_input.shape
        serializable_shape = _SymbolicShape.from_shape_and_range_constraints(
            shape, ep.range_constraints if not _IGNORE_RANGE_CONSTRAINTS else None
        )
        if serializable_shape.is_static_shape():
            ct_inputs.append(
                ct.TensorType(name=name, shape=serializable_shape.shape, dtype=ct_dtype)
            )
            continue
        # Dynamic shape
        assert (
            serializable_shape in emap
        ), f"The shape of input {name} ({serializable_shape}) is not in the _SymbolicShapeToEnumeratedShapeMap={emap}"
        enumerations = emap[serializable_shape]
        ct_enumerated_shape = ct.EnumeratedShapes(shapes=enumerations)
        ct_inputs.append(
            ct.TensorType(name=name, shape=ct_enumerated_shape, dtype=ct_dtype)
        )
    return ct_inputs
