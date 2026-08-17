# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional

# Deliberately stricter than CMake. CMake decides false by exclusion, so anything that is not one
# of its false constants is true, including values like "2.0" and "enabled". Here an unrecognised
# spelling reads as off, because this decides whether a component's libraries are packaged and
# shipping a component whose libraries were never built is worse than shipping one fewer.
_TRUE_VALUES = {"on", "true", "yes", "y"}


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
        """Whether a cache value reads as true, by CMake's rule.

        CMake decides false by exclusion, comparing against its false constants as strings without
        parsing a number. So a bare 0 is false while 0.0 is true, and any word it does not recognise,
        enabled for instance, is true. Two readers in this tree previously used a whitelist of true
        words plus a numeric test, which made an unrecognised word false and disagreed both with CMake
        and with each other. That mismatch let packaging declare no CUDA runtime for a build CMake had
        already turned CUDA on for.
        """
        if value is None:
            return False
        normalized = value.strip().lower()
        if normalized in {"off", "false", "n", "no", "0", "", "ignore", "notfound"}:
            return False
        return not normalized.endswith("-notfound")

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
