# Usage Guide for Models Provided by ExecuTorch

This guide provides examples and instructions for deploying and executing models using the ExecuTorch runtime on Qualcomm platforms.

## Model categories
The following models in two folders can be categorized based on their primary use cases.

1. Language Model:
   - mobilebert_fine_tune

2. Speech Model:
   - wav2letter

3. Vision Model:
   - edsr
   - inception_v3
   - inception_v4
   - mobilenet_v2
   - mobilenet_v3
   - torchvision_vit

## Prerequisite
Please follow another [README](../README.md) first to set up environment.

## Model running
Some models require specific datasets. Please download them in advance and place them in the appropriate folders.

Detailed instructions for each model are provided below.
If you want to export the model without running it, please add `--compile_only` to the command.

1. `deeplab_v3`:
   - Required Dataset : VOCSegmentation  
       Will be downloaded automatically if `-d` is specified

      ```bash
      python deeplab_v3.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} 

2. `edsr`:
   - Required Dataset : DIV2K

       Will be downloaded automatically if -d is specified. Alternatively, you can provide your own dataset using `--hr_ref_dir` and `--lr_dir`.
      
      ```bash
      pip install piq
      python edsr.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} -d 

3. `inception_v3`, `inception_v4`, `mobilenet_v2`,`mobilenet_v3`, `torchvision_vit`:
   - Required Dataset : ImageNet 
       
       Download [dataset](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) first, and place it in a valid folder.
      
      ```bash
      python SCRIPT_NAME.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} -d path/to/ImageNet
 
4. `mobilebert_fine_tune`:
   - You can specify the pretrained weight using `-p <path/to/pretrained_weight>`, if no pretrained weights are provided, using `--num_epochs` to set number of epochs to train the model.
   - `-F --use_fp16`: If specified, the model will run in FP16 mode and the PTQ will be ignored.

      ```bash
      python mobilebert_fine_tune.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} --num_epochs <number_of_epochs>

5.  `wav2letter`:
    - Pretrained weight : 
       
      for torchaudio.models.Wav2Letter version, please download at [here](https://github.com/nipponjo/wav2letter-ctc-pytorch/tree/main?tab=readme-ov-file#wav2letter-ctc-pytorch), and place it in a valid folder.
      ```bash
      python wav2letter.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} -p path/to/pretrained_weight
   