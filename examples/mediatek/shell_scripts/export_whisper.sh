model=${1:-'whisper-large-v3-turbo'}
chunks=${2:-1}
tok=${3:-10}
cache=${4:-512}
cal=${5:-None}
pres=${6:-A16W4}
plat=${7:-DX4}

if [ $model = "whisper-large-v3-turbo" ]
then
	config_path=whisper-large-v3-turbo/config_refactor.json

elif [ $model = "whisper-large-v3" ]
then
	config_path=whisper-large-v3/config_refactor.json
    
fi

if [ $cal = "None" ]
then
	data=""
else
	data="-d aot_utils/mllm_utils/audio/${cal}"
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

python3 model_export_scripts/whisper.py \
    models/llm_models/weights/${config_path} \
    -p $pres \
    --num_chunks $chunks \
	${data} \
    -shapes ${tok}t${cache}c 1t${cache}c \
	--platform $plat
