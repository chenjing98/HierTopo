#!/bin/bash

export CORE_COUNT=50

declare -a nodes=(10 15 20 25 30 35 40 45 50)
declare -a algs=("greedy" "egotree" "bmatch" "oblivious" "dijgreedy")
declare degree=(4)
declare iter=(14)
declare k=(3)
declare -a node_param=(10 20 30)
declare -a iter_param=(10 20 30)
declare -a periods=(66 67)
declare dataset=("scratch")
declare -a adschemes=("replace" "add")

export CURRENT_DIR=$PWD
export RESULT_DIR=${CURRENT_DIR}/../results

export HOP_AVG_REGEX="\[Average Hop\] ([0-9\.e\-]+)"
export HOP_STD_REGEX="\[Standard Deviation Hop\]\ ([0-9\.e\-]+)"
export STEP_AVG_REGEX="\[Average Step\] ([0-9\.e\-]+)"
export STEP_STD_REGEX="\[Standard Deviation Step\]\ ([0-9\.e\-]+)"
export PORT_AVG_REGEX="\[Average Change Port\] ([0-9\.e\-]+)"
export PORT_STD_REGEX="\[Standard Deviation Change Port\]\ ([0-9\.e\-]+)"
export TEST_TIME_REGEX="\[Average Test Time\] ([0-9\.e\-]+) s"

run() {
    n_node=$1
    alg=$2
    filename=$3
    
    output=$(python3 test.py -n "${n_node}" -m "${alg}")
    
    echo "$output" >> "${filename}".txt
    
    if [[ $output =~ $HOP_AVG_REGEX ]]
    then
        hop_avg="${BASH_REMATCH[1]}"
    fi
    
    if [[ $output =~ $HOP_STD_REGEX ]]
    then
        hop_std="${BASH_REMATCH[1]}"
    fi
    
    if [[ $output =~ $TEST_TIME_REGEX ]]
    then
        ttime="${BASH_REMATCH[1]}"
    fi

    if [[ $output =~ $STEP_AVG_REGEX ]]
    then
        step_avg="${BASH_REMATCH[1]}"
    fi

    if [[ $output =~ $STEP_STD_REGEX ]]
    then
        step_std="${BASH_REMATCH[1]}"
    fi

    if [[ $output =~ $PORT_AVG_REGEX ]]
    then
        port_avg="${BASH_REMATCH[1]}"
    fi

    if [[ $output =~ $PORT_STD_REGEX ]]
    then
        port_std="${BASH_REMATCH[1]}"
    fi
    
    echo "${n_node}", "${alg}", "${hop_avg}", "${hop_std}", "${ttime}", "${step_avg}", "${step_std}", "${port_avg}", "${port_std}" >> "${filename}".csv
}

export -f run

filename="${RESULT_DIR}/baseline"

echo Starts at "$(date +"%Y-%m-%d %H:%M:%S")" > ${filename}.txt
echo Starts at "$(date +"%Y-%m-%d %H:%M:%S")" > ${filename}.csv
echo n_node, alg, hop_avg, hop_std, ttime, step_avg, step_std, port_avg, port_std >> ${filename}.csv


parallel -j${CORE_COUNT} run ::: "${nodes[@]}" ::: "${algs[@]}" ::: "${filename}"


echo Ends at "$(date +"%Y-%m-%d %H:%M:%S")" >> "${filename}".txt
echo Ends at "$(date +"%Y-%m-%d %H:%M:%S")" >> "${filename}".csv
