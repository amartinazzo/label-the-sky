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

date=`date "+%y%m%d"`
data_dir=$DATA_PATH

declare -a backbones=(resnext efficientnet vgg)
declare -a outputs=(classes magnitudes)
declare -a nbands=(12 5 3)

declare -a commands
declare -a pids

for backbone in ${backbones[*]}
do
    for n in ${nbands[*]}
    do
        for output in ${outputs[*]}
        do
            commands+=("python -u _exp01.py $data_dir csv/dr1_classes_split.csv $backbone $output $n $date")
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

    for gpu in 0 1
    do
        cmd=${commands[$i]}
        backbone=$(echo $cmd | cut -d" " -f6)
        target=$(echo $cmd | cut -d" " -f7)
        nbandss=$(echo $cmd | cut -d" " -f8)
        
        if [ "$cmd" != "" ]
        then
            logfile="logs/${date}_${backbone}_${target}_${nbandss}.log"
            echo "CUDA_VISIBLE_DEVICES=$gpu $cmd >> $logfile 2>&1 &"
            echo "CUDA_VISIBLE_DEVICES=$gpu $cmd >> $logfile 2>&1 &" >> $logfile
            eval "CUDA_VISIBLE_DEVICES=$gpu $cmd >> $logfile 2>&1 &"
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
