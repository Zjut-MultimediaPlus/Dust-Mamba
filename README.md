# Dust-Mamba
The code of "Dust-Mamba: An Efficient Dust Storm Detection Network With Multiple Data Sources" accepted by AAAI2025.

Paper:....

## Requirements

Python 3.8.8

Pytorch 1.13.0 (GPU)

CUDA 11.7

Dependencies for the Mamba model can be found in the `packs` folder.

## Data Preparation

Dataset: https://github.com/Zjut-MultimediaPlus/LSDSSIMR 

Paper:  https://ieeexplore.ieee.org/abstract/document/10287393

CSV:  Stores the paths for dataset **(already provided)**.

Checkpoints/ **(already provided)**

├── Intensity_Detection/ 

│   ├── +MRDF.pth

│   ├── +VSB.pth

│   ├── AttUNet.pth

│   ├── Dust-Mamba_joint_training.pth

│   ├── Dust-Mamba_train_entire_model.pth

│   ├── Dust-Mamba_train_final_layer.pth   

│   ├── Dust-Mamba_train_from_scratch.pth  

│   ├── SwinUNet.pth            

│   ├── TransUNet.pth          

│   ├── UNet.pth                

│   ├── UNet++.pth              

│   └── VMUNet.pth             

├── Occurrence_Detection/ 

│   ├── 6_Channels/ 

│   │   ├── AttUNet.pth

│   │   ├── SwinUNet.pth

│   │   ├── TransUNet.pth

│   │   ├── UNet.pth 

│   │   └── UNet++.pth

│   └── 18_Channels/ 

│       ├── +MRDF.pth

│       ├── +VSB.pth

│       ├── AttUNet.pth

│       ├── Dust-Mamba.pth

│       ├── SwinUNet.pth

│       ├── TransUNet.pth

│       ├── UNet.pth

│       ├── UNet++.pth

│       └── VMUNet.pth

## Train

### Train from scratch：
python train.py --model=SwinUNet --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/"  --cfg=detect_cfg.yml --save=tmp_dustdetect

### Train transfer learning strategies 1 - 2：
python train_transfer_learning_strategies_1_and_2.py --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/" --dir_pretrained_model="/root/my_model/checkpoint/Occurrence_Detection/18 Channels/Dust-Mamba.pth" --train_all_params=True --cfg=detect_cfg.yml --save=tmp_dustdetect

### Train transfer learning strategies 3：
python train_transfer_learning_strategies_3.py  --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/"  --cfg=detect_cfg.yml --save=tmp_dustdetect


### Test

#### Test model train from scratch or  transfer learning strategies 1 - 2：
python test.py --model=UNet --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/" --dir_checkpoint="/root/my_model/checkpoint/Intensity_Detection/Dust-Mamba_train_from_scratch.pth" --cfg=detect_cfg.yml --save=tmp_dustdetect

#### Test model train from transfer learning strategies 3：

python test_strategies_3.py --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/" --dir_checkpoint="/root/my_model/checkpoint/Intensity_Detection/Dust-Mamba_joint_training.pth" --cfg=detect_cfg.yml --save=tmp_dustdetect
