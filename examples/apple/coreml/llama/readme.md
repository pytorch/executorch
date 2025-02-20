# ANE-friendly Llama models

This directory contains ANE-friendly Llama models.

Export model with:
```
python export.py -n /path/to/output/model.pte -p /path/to/params.json -c /path/to/model.pth --seq_length 64 --max_seq_length 1024 --coreml-quantize c4w
```

The runner is written in python and is only intended to serve as an example for how the model inputs should be processed; it is not performant.


Run model with:
```
python run.py -m /path/to/model.pte -p /path/to/params.json -t /path/to/tokenizer.model --seq_length 64 --max_seq_length 1024 --prompt "Once upon a time," --n_steps 512
```

The model here is based on a "sliding" cache, where old tokens are evicted from the cache.  There is no actual sliding in the implementation, though.tion.


## Export args
* seq_length: the number of tokens processed by the model.  Sequences shorter than seq_length must be padded, and sequences longer than it must be chunked.
* max_seq_length: the maximum context tokens that can be processed.
* cache_size: the size of the KV cache sequences.  This parameter is optional, and defaults to max_seq_length - seq_length.  If a smaller cache_size is used, older tokens are evicted from the cache and no longer play a role in attention.  For example, if max_seq_length=1024, but cache_size is 512, the model can generate up to 1024 tokens, but only the current tokens and the previous 512 will participate in attention.  In terms of computation, cache_size plays a similar role to max_seq_length in models without cache eviction.
* use_cache_list: boolean option that controls whether KV caches are passed as a list of 4D tensors, one per layer, or if they are passed as one 5D tensor.  (Note that use_cache_list does not work with ExecuTorch pybindings.)
* target_size: this option splits linear layers into chunks of target size.  For example, if target_size is 1024, a linear layer with (in_features=512, out_features=8096) will be split into 8 linear layers with (in_features=512, out_features=1024) and the results concatted.  If not specified, the default is no splitting.
* max_splits: this controls the maximum number of splits for linear layers.  It is only relevant if target_size is passed and defaults to 8.

## Llama1B on iPhone 15

We are actively experimenting with different settings, but here are ones we've found that work well on iPhone 15 Pro for Llama1B:
* max_seq_length=1024, seq_length=64, use_cache_list, target_size=1024, max_splits=8
