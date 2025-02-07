#!/bin/bash
set -eou pipefail

trap ctrl_c INT

function ctrl_c() {
  mv trace.txt trace.txt.bak$(date "+%Y.%m.%d-%H.%M.%S")
}

read_timeout=1500 write_timeout=1500 port=31112 ./resmngr "$@"
