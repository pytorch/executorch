This directory contains static, ANE-friendly Llama models.

Export model with:
```
python export.py -n /path/to/output/model.pte -p /path/to/params.json -c /path/to/model.pth --seq_length 64 --max_seq_length 1024 --coreml-quantize c4w
```

Run model with:
```
python run.py -m /path/to/model.pte -p /path/to/params.json -t /path/to/tokenizer.model --seq_length 64 --max_seq_length 1024 --prompt "Once upon a time,"
```

The runner is written in python and is only intended to serve as an example for how the model inputs should be processed; it is not performant.
