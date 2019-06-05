#!/usr/bin/env bash

python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --predictions mask
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.001
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.002
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.005
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.01
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.02
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.05
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.1
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.2
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 0.5
python3 run.py --batch-train=8 --epochs=200 --device=cuda:0 --model=unet --pred-weights 1 1
