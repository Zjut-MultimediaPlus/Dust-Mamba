# Dust-Mamba

<p align="center">
  <strong>
    📄 <a href="https://doi.org/10.1609/aaai.v39i27.34997">Dust-Mamba: An Efficient Dust Storm Detection Network with Multiple Data Sources</a><br>
    <em>Accepted at AAAI 2025</em>
  </strong>
</p>

## Requirements

Python 3.8.8

Pytorch 1.13.0 (GPU)

CUDA 11.7

Dependencies for the Mamba model can be found in the `packs` folder.

## Data Preparation

Dataset: https://github.com/Zjut-MultimediaPlus/LSDSSIMR 

Dataset Paper:  https://ieeexplore.ieee.org/abstract/document/10287393

Checkpoints https://pan.baidu.com/s/1BazQb4v8plLpTbmFODeCbg?pwd=khve

├── Intensity_Detection/ 

├── Occurrence_Detection/ 

│   ├── 6_Channels/ 

│   └── 18_Channels/ 

CSV: Lists the paths of the dataset utilized. **(already provided)**.

## Train


### Train from scratch：
```bash
python train.py --model=SwinUNet --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/"  --cfg=detect_cfg.yml --save=tmp_dustdetect
```


### Train transfer learning strategies 1 - 2：
```bash
python train_transfer_learning_strategies_1_and_2.py --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/" --dir_pretrained_model="/root/my_model/checkpoint/Occurrence_Detection/18 Channels/Dust-Mamba.pth" --train_all_params=True --cfg=detect_cfg.yml --save=tmp_dustdetect
```


### Train transfer learning strategies 3：
```bash
python train_transfer_learning_strategies_3.py  --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/"  --cfg=detect_cfg.yml --save=tmp_dustdetect
```

## Test

### Test model trained from scratch or transfer learning strategies 1 - 2：
```bash
python test.py --model=UNet --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/" --dir_checkpoint="/root/my_model/checkpoint/Intensity_Detection/Dust-Mamba_train_from_scratch.pth" --cfg=detect_cfg.yml --save=tmp_dustdetect
```


### Test model trained from transfer learning strategies 3：
```bash
python test_strategies_3.py --dir_hdfdata="/opt/data/private/" --dir_csv="/opt/data/private/" --dir_checkpoint="/root/my_model/checkpoint/Intensity_Detection/Dust-Mamba_joint_training.pth" --cfg=detect_cfg.yml --save=tmp_dustdetect
```
