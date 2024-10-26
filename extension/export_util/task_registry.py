# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


task_registry = {}


def register_task(task_name):
    def decorator(func):
        task_registry[task_name] = func
        return func

    return decorator
