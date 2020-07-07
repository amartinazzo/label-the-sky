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

declare -a servers=($(hostname))

timestamp=`date "+%y%m%d"`

declare -a backbones=(vgg) #resnext efficientnet)
declare -a outputs=(magnitudes magnitudesmock) #classes)
declare -a nbands_=(12)

declare -a commands
declare -a pids

for nbands in ${nbands_[*]}
do
    for backbone in ${backbones[*]}
    do
        for output in ${outputs[*]}
        do
            commands+=("python -u _exp01.py $backbone $output $nbands $timestamp")
        done
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

    for gpu in 2 3
    do
        cmd=${commands[$i]}
        backbone=$(echo $cmd | cut -d" " -f4)
        target=$(echo $cmd | cut -d" " -f5)
        nbandss=$(echo $cmd | cut -d" " -f6)
        
        if [ "$cmd" != "" ]
        then
            logfile="logs/${timestamp}_${backbone}_${target}_${nbandss}.log"
            echo "TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=$gpu $cmd > $logfile 2>&1 &"
            echo "TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=$gpu $cmd > $logfile 2>&1 &" >> $logfile
            eval "TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=$gpu $cmd > $logfile 2>&1 &"
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
