#!/bin/bash

function waitpids {
    pids=($@)

    echo "waiting on pids:"
    for p in ${pids[*]}
    do
        echo -n "$p "
    done
    echo ""

    wait ${pids[*]}
}

if [ "$1" = "" ]
then
    echo "Usage: $0 <timestamp>"
    exit 1
fi

export timestamp=$1

declare -a servers=($(hostname))

declare -a backbones=(vgg)
declare -a nbands_weights_ft=(
    "3 imagenet 0"
    "3 imagenet 1"
    "3 None 1"
    "12 None 1"
    "3 magnitudes 0"
    "3 magnitudes 1"
    "3 mockedmagnitudes 0"
    "3 mockedmagnitudes 1"
    "12 magnitudes 0"
    "12 magnitudes 1"
    "12 mockedmagnitudes 0"
    "12 mockedmagnitudes 1"
)

declare -a commands
declare -a pids

for backbone in ${backbones[*]}
do
    for arg in "${nbands_weights_ft[@]}"
    do
        commands+=("python -u 02_train.py $backbone $arg $timestamp")
    done
done

OFS=$IFS
IFS=$'\n'

i=0
s=0
gpu=0

while true
do
    server=${servers[$s]}

    for gpu in 0 1 2 3
    do
        cmd=${commands[$i]}
        backbone=$(echo $cmd | cut -d" " -f4)
        nbandss=$(echo $cmd | cut -d" " -f5)
        weights=$(echo $cmd | cut -d" " -f6)
        ft=$(echo $cmd | cut -d" " -f7)
        
        if [ "$cmd" != "" ]
        then
            logfile="logs/${timestamp}_${backbone}_${nbandss}_${weights}_ft${ft}.log"
            echo "CUDA_VISIBLE_DEVICES=$gpu $cmd > $logfile 2>&1 &"
            echo "CUDA_VISIBLE_DEVICES=$gpu $cmd > $logfile 2>&1 &" >> $logfile
            eval "CUDA_VISIBLE_DEVICES=$gpu $cmd > $logfile 2>&1 &"
            pids+=($!)
            i=$((i + 1))
        fi
    done

    s=$((s + 1))

    if [ $s -ge ${#servers[*]} ]
    then
        waitpids ${pids[@]}
        s=0
        pids=()
    fi

    if [ $i -ge ${#commands[*]} ]
    then
        waitpids ${pids[@]}
        break
    fi
done
