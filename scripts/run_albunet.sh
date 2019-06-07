#!/usr/bin/env bash

python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.0005
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.001
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.002
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.005
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.01
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.02
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.05
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.1
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.2
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 0.5
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --pred-weights 1 1
python3 run.py --batch-train=4 --center-factor=90 --epochs=30 --use-dice=true --device=cuda:${1:-0} --wf=5 --model=albunet --pretrained=true --predictions mask
