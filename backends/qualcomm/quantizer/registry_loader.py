# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from functools import lru_cache
from typing import Dict, List, Tuple

from executorch.backends.qualcomm.quantizer.rules import OpQuantRule
from executorch.backends.qualcomm.quantizer.validators import ConstraintCache


# The number of maxsize refers to how many different backends we expect to load.
@lru_cache(maxsize=1)
def load_backend_rules_and_constraints(
    backend: str,
) -> Tuple[Dict[str, List[OpQuantRule]], ConstraintCache]:
    mod = importlib.import_module(
        f"executorch.backends.qualcomm.quantizer.annotators.{backend}_rules"
    )
    rules = mod.get_rules()
    constraints = mod.get_constraint_cache()
    assert isinstance(rules, dict)
    return rules, constraints
