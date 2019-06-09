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

## Install
To use the library from anywhere in python:
```
pip install -e .
```
