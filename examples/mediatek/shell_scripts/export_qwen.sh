model=${1:-'Qwen3-4B'}
chunks=${2:-4}
tok=${3:-128}
cache=${4:-512}
cal=${5:-None}
pres=${6:-A16W4}
plat=${7:-DX4}

if [ $model = "Qwen3-4B" ]
then
	config_path=Qwen3-4B/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/qwen3.json"
elif [ $model = "Qwen3-1.7B" ]
then
	config_path=Qwen3-1.7B/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/qwen3.json"
elif [ $model = "Qwen2-7B-Instruct" ]
then
	config_path=Qwen2-7B-Instruct/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/qwen.json"
elif [ $model = "Qwen2.5-3B" ]
then
	config_path=Qwen2.5-3B/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/qwen.json"
elif [ $model = "Qwen2.5-0.5B-Instruct" ]
then
	config_path=Qwen2.5-0.5B-Instruct/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/qwen.json"
elif [ $model = "Qwen2-1.5B-Instruct" ]
then
	config_path=Qwen2-1.5B-Instruct/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/qwen.json"
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

python3 model_export_scripts/qwen.py \
    models/llm_models/weights/${config_path} \
    -p $pres \
    --num_chunks $chunks \
	${data} \
	${pref} \
    -shapes ${tok}t${cache}c 1t${cache}c \
	--platform $plat
	