# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional

_FALSE_VALUES = {"off", "0", "", "no"}


@dataclass
class CacheValue:
    value_type: str
    value: str


@dataclass
class CMakeCache:
    # The path to the CMakeCache.txt file.
    cache_path: str

    def __post_init__(self):
        self.cache = CMakeCache.read_cmake_cache(cache_path=self.cache_path)

    def get(self, var: str) -> Optional[CacheValue]:
        return self.cache.get(var)

    def is_enabled(self, var: str, fallback: bool = False) -> bool:
        definition = self.get(var)
        if definition is None:
            return fallback
        return CMakeCache._is_truthy(definition.value)

    @staticmethod
    def _is_truthy(value: Optional[str]) -> bool:
        if (value is None) or (value.lower().strip() in _FALSE_VALUES):
            return False
        return True

    @staticmethod
    def read_cmake_cache(cache_path: str) -> Dict[str, CacheValue]:
        result = {}
        with open(cache_path, "r") as cache_file:
            for line in cache_file:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    value_type = ""
                    if ":" in key:
                        key, value_type = key.split(":")
                    result[key.strip()] = CacheValue(
                        value_type=value_type,
                        value=value.strip(),
                    )
        return result
