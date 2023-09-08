#!/bin/bash

#this script is ran in chroot after copy, so take advantage
mkdir -p /cuda_dumps 
rm -f /cuda_dumps/*
mv /function/cuda_dumps/* /cuda_dumps