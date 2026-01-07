#!/usr/bin/env bash

CONTAINER_NAME="owl"
IMAGE_NAME="owl:latest"

if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    docker rm -f "$CONTAINER_NAME"
else
    echo "Container does not exist"
fi

docker run \
  --privileged \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --device /dev/dri \
  -it \
  -e DISPLAY \
  --gpus all \
  -v "$(pwd)/:/workspace" \
  --net host \
  --shm-size 40G \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME" \
  /bin/bash

