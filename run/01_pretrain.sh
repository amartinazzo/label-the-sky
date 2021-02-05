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

dataset="unlabeled"
timestamp=`date "+%m%d"`

declare -a datasets=(unlabeled) #(unlabeled-005-100k unlabeled-01-100k unlabeled-05-100k unlabeled-1-100k)
declare -a backbones=(vgg) #(resnext efficientnet)
declare -a outputs=(magnitudes)
declare -a nbands_=(3 5 12)

declare -a commands
declare -a pids

for backbone in ${backbones[*]} 
do
    for nbands in ${nbands_[*]}
    do
        for output in ${outputs[*]}
        do
            for dataset in ${datasets[*]}
            do
                commands+=("python -u 01_pretrain.py $dataset $backbone $nbands $output $timestamp")
            done
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

    for gpu in 0 1 2 3
    do
        cmd=${commands[$i]}
        dataset=$(echo $cmd | cut -d" " -f4)
        backbone=$(echo $cmd | cut -d" " -f5)
        nbandss=$(echo $cmd | cut -d" " -f6)
        target=$(echo $cmd | cut -d" " -f7)
        
        if [ "$cmd" != "" ]
        then
            logfile="logs/${timestamp}_${dataset}_${backbone}_${target}_${nbandss}.log"
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
