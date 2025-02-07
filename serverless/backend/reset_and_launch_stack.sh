#!/bin/bash
set -euo pipefail

# we need to be in a swarm (master)
function isSwarmNode(){
    if [ "$(docker info | grep Swarm | sed 's/Swarm: //g')" == "inactive" ]; then
        echo false;
    else
        echo true;
    fi
} 

# if not, init
if [ ! "$(isSwarmNode)"] ; then
    docker swarm init
fi

# lets reset
echo "Reseting the docker stack"
docker stack rm func
./deploy_stack.sh