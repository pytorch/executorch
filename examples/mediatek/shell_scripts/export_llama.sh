model=${1:-'llama3'}
chunks=${2:-4}
tok=${3:-128}
cache=${4:-512}

if [ $model = "llama3" ]
then
	config_path=llama3-8B-instruct/config.json
	pres=A16W4
elif [ $model = "llama2" ]
then
	config_path=llama2-7B-chat/config.json
	pres=A16W4
fi

echo "Model: $model"
echo "Config Path: $config_path"
echo "Num Chunks: $chunks"
echo "Num Tokens: $tok"
echo "Cache Size: $cache"
echo "Precision: $pres"

python3 model_export_scripts/llama.py \
    models/llm_models/weights/${config_path} \
    -p $pres \
    --num_chunks $chunks \
    -shapes ${tok}t${cache}c 1t${cache}c
