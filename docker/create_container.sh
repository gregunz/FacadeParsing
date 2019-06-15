#!/usr/bin/env bash
# 9422:22 port if you want to use ssh connection to your docker (e.g. useful for remote interpreter in PyCharm)
# 9488:8888 port to use jupyterlab/jupyter notebook
# ssh keys added to commit on git
# remove/add/change volumes depending on your needs

docker run -d \
  --runtime=nvidia \
  --name ai \
  -p 9422:22 \
  -p 9488:8888 \
  -v ~/projects:/opt/app/projects \
  -v /data/facade_segmentation:/data \
  -v ~/.ssh:/root/.ssh \
  gregunz/jupyterlab:facade_project
