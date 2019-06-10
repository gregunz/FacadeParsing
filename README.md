# Facade Project
Semester Project at Swiss Data Science Center @EPFL

## Description
This project is part of collaboration between civil engineers and the Swiss Data Science Center (SDSC). Its goal is to help civil engineers automating the process of evaluating the damage on a building after an earthquake occurred. This report focuses on a sub-part of this task, which is identifying the building structure, where are the walls, windows, doors, etc.

Find more details about the project in the [report.pdf](report.pdf).

## Dependencies
Please use this docker image which contains every single dependency in order to be sure that each line of code run smoothly.
```
docker pull gregunz/jupyterlab:facade_project
```
It requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and CUDA 10.0.

Find in directory [docker](docker), scripts to start your docker environment.

## Install
To install the `facade_project` library and use it within your python environment:
```
pip install -e .
```
where `.` is the root directory of this git repository, containing the installer (`setup.py` file).

Note that the Docker container already includes the library.

## How-to-use
Find use examples in [notebooks](notebooks) and [scripts](scripts).

To train, the `scripts/run.py` script is ready to use:
```
usage: run.py [-h] [--model MODEL] [--epochs EPOCHS] [--split-seed SPLIT_SEED]
              [--batch-train BATCH_TRAIN] [--batch-val BATCH_VAL] [--wf WF]
              [--pretrained PRETRAINED]
              [--predictions PREDICTIONS [PREDICTIONS ...]]
              [--pred-weights PRED_WEIGHTS [PRED_WEIGHTS ...]]
              [--path-for-weights PATH_FOR_WEIGHTS] [--device DEVICE]
              [--use-dice USE_DICE] [--crop-size CROP_SIZE]
              [--center-factor CENTER_FACTOR]
              [--load-trained-name LOAD_TRAINED_NAME]
              [--load-trained-epoch LOAD_TRAINED_EPOCH]

Script to perform facade parsing

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --epochs EPOCHS
  --split-seed SPLIT_SEED
  --batch-train BATCH_TRAIN
  --batch-val BATCH_VAL
  --wf WF
  --pretrained PRETRAINED
  --predictions PREDICTIONS [PREDICTIONS ...]
  --pred-weights PRED_WEIGHTS [PRED_WEIGHTS ...]
  --path-for-weights PATH_FOR_WEIGHTS
  --device DEVICE
  --use-dice USE_DICE
  --crop-size CROP_SIZE
  --center-factor CENTER_FACTOR
  --load-trained-name LOAD_TRAINED_NAME
  --load-trained-epoch LOAD_TRAINED_EPOCH
```
For example:
```
python3 run.py --batch-train=4 --epochs=25 --use-dice=true --device=cuda:0 --model=albunet --pretrained=true --pred-weights 1 0.002
```

For predictions, one should look into `notebooks/nb_predictions.ipynb`


Few high-level details can also be found in the [report.pdf](report.pdf)
