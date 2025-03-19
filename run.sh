for i in {1..5}; do
    ./cmake-out/examples/models/llama/llama_main --model_path=$MODEL_OUT --tokenizer_path=$TOKENIZER --prompt="Once upon a time,"
done
