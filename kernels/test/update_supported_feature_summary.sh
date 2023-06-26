#!/bin/bash

buck run fbcode//executorch/kernels/test:summarize_supported_features > fbcode/executorch/kernels/test/supported_features_summary.md
