#!/usr/bin/env bash

docker run -d \
  --runtime=nvidia \
  --name ai \
  -p 9422:22 \
  -p 9488:8888 \
  -v ~/projects:/opt/app/projects \
  -v /data/facade_segmentation:/data \
  -v ~/.ssh:/root/.ssh \
  gregunz/jupyterlab:facade_project
