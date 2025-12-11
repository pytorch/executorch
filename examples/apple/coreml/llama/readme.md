# ANE-friendly Llama models

To export a static, ANE-friendly model use:

```
python export_static_llm_coreml.py \
    --checkpoint /path/to/model.pth \
    --params /path/to/params.json \
    --output static_llm_coreml_model.pte
```

To test in python, use:

```
python run_static_llm.py \
    --model static_llm_coreml_model.pte \
    --params /path/to/params.json \
    --tokenizer /path/to/tokenizer.model \
    --prompt "Once upon a time" \
    --max_new_tokens 100 \
    --lookahead
```

(Enabling lookahead decoding is optional, but does improve performance.)

The static model has several ANE optimizations, including:
* Splitting linear layers for improved performance (controlled by target_split_size and max_splits args)
* Splitting the pte into multiple Core ML pieces for improved performance (can be disabled with no_graph_breaks)
* Re-writing SDPA to avoid 5-D tensors to imporve performance.  This also fixes an accuracy bug that was introduced in iOS 26 (addresses this: https://github.com/pytorch/executorch/issues/15833)


We are working on adding a C++ runner as well.


# Deprecated (export.py, run.py, and run_lookahead.py)

Below we describe export.py, run.py, and run_lookahead.py.  But these are deprecated and will evenutally be removed because we are unifying around the static model formulation.

This directory contains ANE-friendly Llama models.

Export model with:
```
python export.py -n /path/to/output/model.pte -p /path/to/params.json -c /path/to/model.pth --seq_length 64 --max_seq_length 1024 --coreml-quantize c4w --dtype fp16
```

(Note the script should be run from the executorch/examples/apple/coreml/llama directory.)

The runner is written in python and is only intended to serve as an example for how the model inputs should be processed; it is not performant.


Run model with:
```
python run.py -m /path/to/model.pte -t /path/to/tokenizer.model --prompt "Once upon a time,"
```

The runner can also be used to run an eager model model to compare with CoreML numerics (--use_eager).  In this case, you must specify:
* --checkpoint
* --dtype
* --max_seq_length
* --seq_length

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

* Set use_cache_list.
* Use seq_length = 32, which offers a good balance between prefill/decode performance.
* Split out_features in linear layers with target_split_size=1024, max_splits=8.
* For ANE, set dtype = fp16, coreml-quantize = c4w.  The requires doing QAT on Llama1B for good accuracy.
* Set embedding-quantize to "4,32".
* Set max_seq_length to 128, 256, 512, 1024, and 2048, depending on needed context.  Note that performance drops with max_seq_length.  More specifically, performance drops with cache_size, and the best experience may require a good cache eviction policy.  The python runner in run.py uses a last-in-last-out policy when cache_size is specified.
