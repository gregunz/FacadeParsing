#!/usr/bin/env bash

# different weights for the loss
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.0001
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.0002
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.0005
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.001
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.002
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.005
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.01
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.02
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.05
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.1
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.2
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 0.5
python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --pred-weights 1 1

# only mask or heatmaps
# python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --predictions heatmaps
# python3 run.py --epochs=50 --device=cuda:${1:-0} --model=${2:-albunet} --batch-train=${3:-4} --predictions mask
