# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This removes the imported [examples] module from pip installing lm_eval due to
colliding module name with ET package
"""

try:
    # If the import fails, this means there's nothing to remove
    import examples

    try:
        # If the import succeeds, this means it isn't using lm_eval's module
        import examples.models
    except:
        print(
            "Failed to import examples.models due to lm_eval conflict. Removing lm_eval examples module"
        )
        import shutil

        examples_path = examples.__path__[0]
        shutil.rmtree(examples_path)

except:
    pass
