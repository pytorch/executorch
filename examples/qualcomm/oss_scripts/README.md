# Usage Guide for Models Provided by ExecuTorch

This guide provides examples and instructions for open source models. Some models under this folder might also have their own customized runner.

## Model categories
The following models can be categorized based on their primary use cases.

1. Language Model:
   - albert
   - bert
   - distilbert
   - eurobert
   - llama
   - roberta

2. Vision Model:
   - conv_former
   - cvt
   - deit
   - dino_v2
   - dit
   - efficientnet
   - efficientSAM
   - esrgan
   - fastvit
   - fbnet
   - focalnet
   - gMLP_image_classification
   - mobilevit1
   - mobilevit_v2
   - pvt
   - regnet
   - retinanet
   - squeezenet
   - ssd300_vgg16
   - swin_transformer

## Prerequisite
Please follow another [README](../README.md) first to set up environment.

## Model running
Some models require specific datasets. Please download them in advance and place them in the appropriate folders.

Detailed instructions for each model are provided below.
If you want to export the model without running it, please add `--compile_only` to the command.

1. `albert`,`bert`,`distilbert`, `eurobert`, `roberta`:
   - Required Dataset : wikisent2 
       
      download [dataset](https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences) first, and place it in a valid folder.
      ```bash
      python albert.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} -d path/to/wikisent2

2. `conv_former`,`cvt`,`deit`,`dino_v2`,`efficientnet`,`fbnet`, `focalnet`, `gMLP_image_classification`,  `mobilevit1`,`mobilevit_v2`, `pvt`, `squeezenet`, `swin_transformer` :
   - Required Dataset : ImageNet 
       
      Download [dataset](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) first, and place it in a valid folder.
      ```bash
      python SCRIPT_NAME.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} -d path/to/ImageNet

3. `dit`:

      ```bash
        python dit.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} 
4. `esrgan`:
    - Required Dataset: B100

      Will be downloaded automatically if -d is specified. Alternatively, you can provide your own dataset using `--hr_ref_dir` and `--lr_dir`.

    - Required OSS Repo: Real-ESRGAN

      Clone [OSS Repo](https://github.com/ai-forever/Real-ESRGAN) first, and place it in a valid folder. 
    
      ```bash
      python esrgan.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} --oss_repo path/to/Real-ESRGAN

5. `fastvit`:
    - Required Dataset: ImageNet

      Download [dataset](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) first, and place it in a valid folder.
    
    - Required OSS Repo: ml-fastvit

      Clone [OSS Repo](https://github.com/apple/ml-fastvit) first, and place it in a valid folder.
    
    - Pretrained weight: 

        Download [pretrained weight](https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_s12_reparam.pth.tar) first, and place it in a valid folder(should be fastvit_s12_reparam.pth.tar).
      ```bash
      python fastvit.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} --oss_repo path/to/ml-fastvit -p path/to/pretrained_weight -d path/to/ImageNet

6. `regnet`:
     - Required Dataset: ImageNet

        Download [dataset](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) first, and place it in a valid folder.   
     - Weights: regnet_y_400mf, regnet_x_400mf

        use `--weights` to specify which regent weights/model to execute.
    ```bash
      python regnet.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} -d path/to/ImageNet --weights <WEIGHTS>

7. `retinanet`:
    - Required Dataset: COCO

      Download [val2017](http://images.cocodataset.org/zips/val2017.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) first, and place it in a valid folder.

      ```bash
      python retinanet.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} -d path/to/PATH/TO/COCO #(which contains 'val_2017' & 'annotations')
      
8. `ssd300_vgg16`:
    - Required OSS Repo: 

      Clone [OSS Repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) first, and place it in a valid folder.   
      
    - Pretrained weight: 

        Download [pretrained weight](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) first, and place it in a valid folder.(checkpoint_ssd300.pth.tar)   

   - Required Dataset: VOCSegmentation  
      download [VOC 2007](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection?tab=readme-ov-file#download) first, and place it in a valid folder.
      ```bash
      python ssd300_vgg16.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} --oss_repo path/to/a-PyTorch-Tutorial-to-Object-Detection -p path/to/pretrained_weight 

9. `llama`:
    For llama, please check [README](llama/README.md) under llama folder for more details.

10. `efficientSAM`:
    For efficientSAM, please get access to efficientSAM folder.
    - Pretrained weight: 

        Download [EfficientSAM-S](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vits.pt.zip) or [EfficientSAM-Ti](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vitt.pt) first, and place it in a valid folder.
     - Required Dataset: ImageNet

        Download [dataset](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) first, and place it in a valid folder. 

    - Required OSS Repo: 

      Clone [OSS Repo](https://github.com/yformer/EfficientSAM) first, and place it in a valid folder.
      ```bash
      python efficientSAM.py -m ${SOC_MODEL} -b path/to/build-android/ -s ${DEVICE_SERIAL} --oss_repo path/to/EfficientSAM -p path/to/pretrained_weight -d path/to/ImageNet 
    