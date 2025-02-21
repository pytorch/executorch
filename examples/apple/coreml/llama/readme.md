# ANE-friendly Llama models

This directory contains ANE-friendly Llama models.

Export model with:
```
python export.py -n /path/to/output/model.pte -p /path/to/params.json -c /path/to/model.pth --seq_length 64 --max_seq_length 1024 --coreml-quantize c4w
```

(Note the script should be run from the executorch/examples/apple/coreml/llama directory.)

The runner is written in python and is only intended to serve as an example for how the model inputs should be processed; it is not performant.


Run model with:
```
python run.py -m /path/to/model.pte -t /path/to/tokenizer.model --prompt "Once upon a time,"
```

(Note the script should be run from the executorch/examples/apple/coreml/llama directory.)


## Export args
* seq_length: the number of tokens processed by the model.  Sequences shorter than seq_length must be padded, and sequences longer than it must be chunked.
* max_seq_length: the maximum context tokens that can be processed.
* cache_size: the size of the KV cache sequences.  This parameter is optional, and defaults to max_seq_length - seq_length.  If a smaller cache_size is used, older tokens are evicted from the cache and no longer play a role in attention.  For example, if max_seq_length=1024, but cache_size is 512, the model can generate up to 1024 tokens, but only the current tokens and the previous 512 will participate in attention.  In terms of computation, cache_size plays a similar role to max_seq_length in models without cache eviction.
* use_cache_list: boolean option that controls whether KV caches are passed as a list of 4D tensors, one per layer, or if they are passed as one 5D tensor.  (Note that use_cache_list does not work with ExecuTorch pybindings.)
* target_split_size: this option splits linear layers into chunks of target size.  For example, if target_split_size is 1024, a linear layer with (in_features=512, out_features=8096) will be split into 8 linear layers with (in_features=512, out_features=1024) and the results concatted.  If not specified, the default is no splitting.
* max_splits: this controls the maximum number of splits for linear layers.  It is only relevant if target_size is passed and defaults to 8.

## Llama1B on iPhone 15

We are actively experimenting with different settings.  But here are ones that we've found work well for Llama1B on iPhone 15 Pro:

* Set use_cache_list
* Split linear layers with target_split_size=1024, max_splits=8
* Use seq_length=32 or seq_length=64, both of which offer reasonable tradeoffs for prefill and decode performance.  seq_length=32 is better at decode and seq_length=64 is better at prefill.

In our tests, we set max_seq_length=1024, but if your application allows for it, performance can improve with max_seq_length=512 or by keeping max_seq_length=1024 and setting cache_size=512-seq_length.
