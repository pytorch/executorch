model=${1:-'llama3.2-3b'}
chunks=${2:-4}
tok=${3:-128}
cache=${4:-512}
cal=${5:-None}
pres=${6:-A16W4}
plat=${7:-DX4}

if [ $model = "llama3.2-3b" ]
then
	config_path=Llama-3.2-3B-Instruct/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/llama3.json"
elif [ $model = "llama3.2-1b" ]
then
	config_path=Llama-3.2-1B-Instruct/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/llama3.json"
elif [ $model = "llama3" ]
then
	config_path=llama3-8B-instruct/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/llama3.json"
elif [ $model = "llama2" ]
then
	config_path=llama2-7B-chat/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/llama2_short.json"
fi

if [ $cal = "None" ]
then
	data=""
else
	data="-d aot_utils/llm_utils/prompts/${cal}"
fi

echo "Model: $model"
echo "Config Path: $config_path"
echo "Num Chunks: $chunks"
echo "Num Tokens: $tok"
echo "Cache Size: $cache"
echo "Precision: $pres"
echo "Calibration Dataset: $cal"
echo "Preformatter: $pref"
echo "Platform: $plat"

python3 model_export_scripts/llama.py \
    models/llm_models/weights/${config_path} \
    -p $pres \
    --num_chunks $chunks \
	${data} \
	${pref} \
    -shapes ${tok}t${cache}c 1t${cache}c \
	--platform $plat
	