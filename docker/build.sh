#!/usr/bin/env bash
#
CONTAINER_NAME="owl"
IMAGE_NAME="owl:latest"


docker build -f docker/owl.docker_container -t $IMAGE_NAME .
