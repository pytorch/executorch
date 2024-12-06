rem Install snakeviz for cProfile flamegraph
rem Install sentencepiece for llama tokenizer
pip install snakeviz sentencepiece

rem Install torchao.
pip install "%~dp0/../../../third-party/ao"

rem Install lm-eval for Model Evaluation with lm-evalution-harness
rem Install tiktoken for tokenizer
pip install lm_eval==0.4.5
pip install tiktoken blobfile
rem Restore numpy if >= 2.0
pip install "numpy<2.0"

rem Call the install helper for further setup
python examples/models/llama/install_requirement_helper.py
