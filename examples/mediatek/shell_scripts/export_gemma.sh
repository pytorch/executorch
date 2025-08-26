model=${1:-'gemma2_2b_it'}
chunks=${2:-4}
tok=${3:-128}
cache=${4:-512}
cal=${5:-None}
pres=${6:-A16W4}

if [ $model = "gemma2_2b_it" ]
then
	config_path=gemma2_2b_it/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/gemma.json"
elif [ $model = "gemma3_1b_it" ]
then
	config_path=gemma3_1b_it/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/gemma.json"
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

python3 model_export_scripts/gemma.py \
    models/llm_models/weights/${config_path} \
    -p $pres \
    --num_chunks $chunks \
	${data} \
	${pref} \
    -shapes ${tok}t${cache}c 1t${cache}c