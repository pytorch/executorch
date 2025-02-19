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

The model here is based on a "sliding" cache, where old tokens are evicted from the cache.  By default, the cache size is max_seq_length - seq_length, but you can explicitly pass in a smaller cache size (e.g., --cache_size 512).  This can speed up computation and reduce memory.  Keep in mind that once cache_size is reached, older tokens get evicted from the cache and do not participate in attention.
