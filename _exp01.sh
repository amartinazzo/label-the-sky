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

# if [ "$1" = "" ]
# then
#     echo "Usage: $0 <timestamp>"
#     exit 1
# fi

date=`date "+%y%m%d"`
data_dir=$DATA_PATH

declare -a outputs=(classes magnitudes)
declare -a nbands=(12 5 3)
declare -a feature_dims=(1024) #(512 256 128 64)

declare -a commands
declare -a pids

for feature_dim in ${feature_dims[*]}
do
    for n in ${nbands[*]}
    do
        for output in ${outputs[*]}
        do
            commands+=("python -u _exp01.py $data_dir csv/dr1_classes_split.csv $output $n $feature_dim")
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

    for gpu in 0
    do
        cmd=${commands[$i]}
        task=$(echo $cmd | cut -d" " -f6)
        nbands=$(echo $cmd | cut -d" " -f7)
        feat_dim=$(echo $cmd | cut -d" " -f8)
        
        if [ "$cmd" != "" ]
        then
            logfile="logs/${date}_${task}_${nbands}bands_${feat_dim}.log"
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
