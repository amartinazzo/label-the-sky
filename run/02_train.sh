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

dataset="clf"

declare -a servers=($(hostname))

declare -a backbones=(vgg)
declare -a pretraining_datasets=(unlabeled imagenet None)
declare -a nbands_=(12 5 3)
declare -a finetune=(0 1)
declare -a dataset_modes=(lowdata full)

declare -a commands
declare -a pids

for backbone in ${backbones[*]}
do
    for nbands in ${nbands_[*]}
    do
        for pretraining_data in ${pretraining_datasets[*]}
        do
            for ft in ${finetune[*]}
            do
                for dataset_mode in ${dataset_modes[*]}
                do
                    if [[ $pretraining_data == "imagenet" && $nbands -ne 3 ]] || [[ $pretraining_data == "None" && $ft -eq 1 ]]
                    then
                        continue
                    else
                        commands+=("python -u 02_train.py $dataset $backbone $pretraining_data $nbands $ft $dataset_mode $timestamp")
                    fi
                done
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
        weights=$(echo $cmd | cut -d" " -f6)
        nbandss=$(echo $cmd | cut -d" " -f7)
        ft=$(echo $cmd | cut -d" " -f8)
        datasetmode=$(echo $cmd | cut -d" " -f9)

        if [ "$cmd" != "" ]
        then
            logfile="logs/${timestamp}_${dataset}_${backbone}_${nbandss}_${weights}_ft${ft}_${datasetmode}.log"
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
