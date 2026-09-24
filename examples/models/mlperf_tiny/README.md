# MLPerf Tiny Streaming Wakeword

`StreamingWakeWord` implements the [MLCommons reference network](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/streaming_wakeword/keras_model.py)
with four incremental depthwise/pointwise convolution blocks. Temporal kernel
sizes are 3, 5, 10, and 15; output channels are 128, 128, 128, and 32. A final
linear layer produces scores for Marvin, silence, and other, in that order.
The model returns logits by default; `apply_softmax=True` returns probabilities.

Each call accepts one FP32 LFBE feature frame with logical NCHW shape
`[1, 40, 1, 1]`. Audio feature extraction is external to this model. The reference
frontend uses 16 kHz audio, a 64 ms window, and a 32 ms hop. The example uses
randomly initialized weights; pretrained weights and accuracy evaluation are
not included.

The four mutable buffers retain 2, 4, 9, and 14 frames of each block's input.
They occupy 14,144 bytes in FP32. Temporary tensors and weights require
additional memory. The first 29 outputs after initialization or reset are
warm-up outputs. Starting with the 30th frame, inference is equivalent to
evaluating the reference network on the latest 30 frames. A shorter stream
has no valid output.

The convolutions and classifier require 46,040 MACs per frame, versus 826,368
for recomputing a complete 30-frame window. These counts exclude normalization,
activations, softmax, and data movement.

## Eager inference

Use `.eval()` and `torch.no_grad()`, and call `reset()` between independent
streams. Each concurrent stream needs its own model instance.

```python
import torch
from executorch.examples.models.mlperf_tiny import StreamingWakeWord

model = StreamingWakeWord(apply_softmax=True).eval()
feature_frames = torch.randn(100, *model.FRAME_SHAPE)

with torch.no_grad():
    for index, frame in enumerate(feature_frames):
        probabilities = model(frame)
        if index >= model.RECEPTIVE_FIELD - 1:
            print(probabilities)
    model.reset()
```

The example model factory registers this model as `streaming_wakeword` and
returns an evaluation model with softmax enabled.

## Tests

```bash
OMP_NUM_THREADS=1 python -m pytest examples/models/test/test_streaming_wakeword.py
```

The tests compare streaming inference with complete 30-frame windows for random,
zero, impulse, and alternating inputs, and cover reset/replay, independent
streams, the model factory, and the single-frame input contract.
