#!/usr/bin/env bash
set -e  # exit on error

if [ -z "$1" ]
  then
    echo "Please specify docker image name as the first argument"
    echo "Usage $0 DOCKER_IMAGE_NAME"
    exit 1
fi

DOCKER_IMAGE=${1}
shift # Consume argument 1
RUN_DOCKER_INTERACTIVE=${RUN_DOCKER_INTERACTIVE:-1}
ROOT_DIR=$(cd "$(dirname "$0")"/../../; pwd)
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/$DOCKER_IMAGE}
if [[ -z "${DOCKER_USER}" ]]; then
  DOCKER_USER="$(id -u):$(id -g)"
fi

DEBUG_FLAGS="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
DOCKER_MAP="-v $PWD:$PWD -w $PWD -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group -v \
  $ROOT_DIR:/source -v $CACHE_DIR/ccache:/root/.ccache -v $HOME/.ssh:/root/.ssh:ro"

DOCKER_FLAGS="--rm ${DOCKER_MAP} --user $(id -u):$(id -g) --ipc=host --security-opt seccomp=unconfined ${DEBUG_FLAGS}"
DOCKER_FLAGS="${DOCKER_FLAGS} --gpus all --network host --mount type=tmpfs,destination=/tmpfs"

if [ "${RUN_DOCKER_INTERACTIVE}" -eq 1 ]; then
    DOCKER_CMD="docker run -i -t ${DOCKER_FLAGS} ${DOCKER_IMAGE} /bin/zsh"
else
    DOCKER_CMD="docker run -t -i ${DOCKER_FLAGS} ${DOCKER_IMAGE}"
fi

${DOCKER_CMD} "$@"
