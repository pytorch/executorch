model=$1

echo "Export model: $model"

if [ $model = "deeplabv3" ]
then
	python3 model_export_scripts/deeplab_v3.py -d
elif [ $model = "edsr" ]
then
	python3 model_export_scripts/edsr.py -d
elif [ $model = "inceptionv3" ]
then
	python3 model_export_scripts/inception_v3.py -d PATH_TO_DATASET
elif [ $model = "inceptionv4" ]
then
	python3 model_export_scripts/inception_v4.py -d PATH_TO_DATASET
elif [ $model = "mobilenetv2" ]
then
	python3 model_export_scripts/mobilenet_v2.py -d PATH_TO_DATASET
elif [ $model = "mobilenetv3" ]
then
	python3 model_export_scripts/mobilenet_v3.py -d PATH_TO_DATASET
elif [ $model = "resnet18" ]
then
	python3 model_export_scripts/resnet18.py -d PATH_TO_DATASET
elif [ $model = "resnet50" ]
then
	python3 model_export_scripts/resnet50.py -d PATH_TO_DATASET
elif [ $model = "dcgan" ]
then
	python3 model_export_scripts/dcgan.py
elif [ $model = "wav2letter" ]
then
	python3 model_export_scripts/wav2letter.py
elif [ $model = "vit_b_16" ]
then
	python3 model_export_scripts/vit_b_16.py
elif [ $model = "mobilebert" ]
then
	python3 model_export_scripts/mobilebert.py
elif [ $model = "emformer_rnnt" ]
then
	python3 model_export_scripts/emformer_rnnt.py
elif [ $model = "bert" ]
then
	python3 model_export_scripts/bert.py
elif [ $model = "distilbert" ]
then
	python3 model_export_scripts/distilbert.py
fi
