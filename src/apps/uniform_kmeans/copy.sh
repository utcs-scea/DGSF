#!/bin/bash

counter=2
until [ $counter -gt 50 ]
do
    cp -r kmeans1/ kmeans$counter
    ((counter++))
done
echo All done
