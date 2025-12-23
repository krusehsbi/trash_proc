#!/bin/bash

# Number of repetitions
N=500

for ((i=1; i<=N; i++)); do
    echo "Run $i / $N"
    blenderproc run trash_proc_coneyor.py
done
