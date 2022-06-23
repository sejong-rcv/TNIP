#------Setting------#
# before make container, Enter the following information.

IMAGE_HEADER = 
BASE_CONDA_ENV_NAME = 
CONTAINER_NAME = 

IMAGE_NAME = tnip
IMAGE_TAG = torch1.10.0-cuda10.2-cudnn8-ubuntu18.04
PYTHON_VERSION = 3.7
UID := $(shell id -u)
USER_NAME := $(shell whoami)
POS_GPUS = all

docker-base:
	@nvidia-docker build -t ${IMAGE_HEADER}/${IMAGE_NAME}:${IMAGE_TAG} \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg CONDA_ENV_NAME=${BASE_CONDA_ENV_NAME} \
		--build-arg UID=${UID} \
		--build-arg USER_NAME=${USER_NAME} \
		-f ./Dockerfile \
		.

docker-run:
	@nvidia-docker run -it -u ${USER_NAME} \
		--name ${CONTAINER_NAME} \
		-w /home/${USER_NAME} \
		-v /home:/home \
		-e NVIDIA_VISIBLE_DEVICES=${POS_GPUS} \
		--shm-size=32G \
		--ipc=host \
		${IMAGE_HEADER}/${IMAGE_NAME}:${IMAGE_TAG} \
		/bin/bash 
