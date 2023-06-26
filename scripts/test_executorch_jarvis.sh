#!/usr/bin/env bash

buck run @fbcode//on_device_ai/Assistant/Jarvis/mode/Harmony_HiFi4_Opus_Tie_5/dev-eh \
    fbcode//on_device_ai/Assistant/Jarvis/min_runtime:min_runtime_size_test -- \
    fbcode/executorch/test/models/linear_out.ff
