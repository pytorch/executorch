model=${1:-'phi-4-mini-reasoning'}
chunks=${2:-4}
tok=${3:-128}
cache=${4:-512}
cal=${5:-None}
pres=${6:-A16W4}

if [ $model = "phi3.5-mini-instruct" ]
then
	config_path=phi3.5-mini-instruct/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/phi3.json"
elif [ $model = "phi-4-mini-reasoning" ]
then
	config_path=phi-4-mini-reasoning/config.json
	pref="--preformatter aot_utils/llm_utils/preformatter_templates/phi3.json"
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

python3 model_export_scripts/phi.py \
    models/llm_models/weights/${config_path} \
    -p $pres \
    --num_chunks $chunks \
	${data} \
	${pref} \
    -shapes ${tok}t${cache}c 1t${cache}c
