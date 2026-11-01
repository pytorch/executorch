# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Record op cases into .ptn packages with bundled samples.

A package holds one method per case and dtype under one quantization scheme,
and two sample sets over the same inputs: `golden`, the eager outputs, and
`precise`, the same math in float64 with quantized weights decoded in float32,
whatever the method's dtype, and activations unquantized, floating outputs
stored as float32. Weight-only schemes add `golden-int8_act`, the eager outputs
with the same weights and int8 per-token dynamic activations. Nothing in it
depends on an engine or a device: consumers judge an engine by its distance
from `precise` relative to eager's.
"""

import copy
import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from executorch.backends.native.test.samples.cases import Case, Inputs, outputs
from executorch.backends.native.test.samples.schemes import int8_activations, Scheme
from executorch.exir.native import Call, Sample, SampleSet, to_native, write_samples
from torch.func import functional_call

logger: logging.Logger = logging.getLogger(__name__)

DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


class _Method(torch.nn.ModuleDict):
    """One model keyed by its method name, keeping constant FQNs unique across a
    package's methods."""

    def forward(self, *args):
        (model,) = self.values()
        return model(*args)


def _cast(inputs: Inputs, dtype: torch.dtype) -> Inputs:
    return tuple(t.to(dtype) if t.is_floating_point() else t for t in inputs)


def record(
    cases: list[Case],
    scheme: Scheme,
    path: Path,
    dtypes: Iterable[str] = tuple(DTYPES),
) -> None:
    """Writes no package, removing any at `path`, when the scheme fits no case."""
    programs, golden, precise, golden_act = {}, [], [], []
    for case in cases:
        if not scheme.fits(case.model):
            logger.info(f"{path.name}: {case.name} does not fit the scheme")
            continue
        for suffix in dtypes:
            name = f"{case.name}_{suffix}"
            if name in programs:
                raise ValueError(f"{path.name}: duplicate method {name}")
            dtype = DTYPES[suffix]
            model = copy.deepcopy(case.model).to(dtype).eval()
            scheme.quantize(model)
            method = _Method({name: model})
            state = {
                k: scheme.decode(v).double() for k, v in method.state_dict().items()
            }
            inputs = [_cast(x, dtype) for x in case.inputs]
            programs[name] = torch.export.export(method, inputs[0])
            with torch.no_grad():
                for x in inputs:
                    golden.append(Sample([Call(name, list(x), outputs(method(*x)))]))
                    ref = functional_call(method, state, _cast(x, torch.float64))
                    ref = [
                        t.float() if t.is_floating_point() else t for t in outputs(ref)
                    ]
                    precise.append(Sample([Call(name, list(x), ref)]))
                    if scheme.weight_only:
                        with int8_activations(method):
                            golden_act.append(
                                Sample([Call(name, list(x), outputs(method(*x)))])
                            )
    if not programs:
        logger.info(f"{path.name}: no case fits the scheme, not written")
        path.unlink(missing_ok=True)
        return
    to_native(programs).save(str(path))
    source = f"torch {torch.__version__}"
    write_samples(str(path), "golden", SampleSet(golden, "eager", source))
    write_samples(
        str(path),
        "precise",
        SampleSet(
            precise,
            "float64, quantized weights decoded in float32, activations unquantized",
            source,
        ),
    )
    if scheme.weight_only:
        write_samples(
            str(path),
            "golden-int8_act",
            SampleSet(golden_act, "eager, int8 per-token dynamic activations", source),
        )
