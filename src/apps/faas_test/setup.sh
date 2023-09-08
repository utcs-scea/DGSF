#!/bin/bash

apt -y install llvm-9
printf "\n\nPATH=\"${PATH}:/usr/lib/llvm-9/bin\"\n" >> /etc/profile
ldconfig
cd /usr/bin && ln -s llvm-config-9 llvm-config
