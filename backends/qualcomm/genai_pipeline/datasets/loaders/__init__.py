# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Purpose-agnostic dataset loaders and collators, organized by modality.

``llm/`` and ``mllm/`` each provide the data-source loaders (lm_eval tasks,
message-sample JSON) and the component-aware collator for that modality. The
purpose (calibration / training / evaluation) is layered on top by the adapter
packages, not here.
"""
