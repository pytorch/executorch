from dataclasses import dataclass

"""
Please refer to executorch/schema/schema.fbs for source of truth.
"""


@dataclass
class CompileSpec:
    key: str  # like max_value
    value: bytes  # like 4 or other types
