#!/bin/bash

function waitpids {
    pids=($@)

    echo "Waiting on pids:"
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
#     echo "Usage: $0 <output_dir>"
#     exit 1
# fi

declare -a outputs=(classes magnitudes redshift)
declare -a nbands=(12 5 3)

declare -a commands
declare -a pids

export output_dir=$DATA_PATH/label_the_sky

for output in ${outputs[*]}
do
    for nband in ${nbands[*]}
    do
        commands+=("python classifiers/image_classifier.py $output $nbands")
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
        task=$(echo $cmd | cut -d" " -f3)
        nbands=$(echo $cmd | cut -d" " -f4)
        
        if [ "$cmd" != "" ]
        then
            logfile="logs/${server}_${task}_${nbands}bands.log"
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
