#!/bin/bash

export CORE_COUNT=40

declare -a nodes=(28 30 35 40 45 50)
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
    n_degree=$2
    n_iter=$3
    n_node_param=$4
    n_iter_param=$5
    period=$6
    dataset_mode=$7
    ad_scheme=$8
    k_order=$9
    filename=${10}
    core_count=${11}
    if [[ ${ad_scheme} == "add" ]]
    then
        file_solution="../poly_log/log${n_node_param}_${n_degree}_${k_order}_${n_iter_param}_same.pkl"
    else
        file_solution="../poly_log/log${n_node_param}_${n_degree}_${k_order}_${n_iter_param}_same_repl.pkl"
    fi
    
    if [[ ! -f "${file_solution}" ]]
    then
        return 1
    fi
    
    output=$(python3 safehiertopo.py -n "${n_node}" -d "${n_degree}" -i "${n_iter}" -np "${n_node_param}" -ip "${n_iter_param}" -p "${period}" -k "${k_order}" -ds "${dataset_mode}" -a "${ad_scheme}" -c "${core_count}" -seq)
    
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
    
    echo "${n_node}", "${n_iter}", "${n_node_param}", "${n_iter_param}", "${period}", "${ad_scheme}", "${hop_avg}", "${hop_std}", "${ttime}", "${step_avg}", "${step_std}", "${port_avg}", "${port_std}" >> "${filename}".csv
}

export -f run

filename="${RESULT_DIR}/fallback_sequential_new"

echo Starts at "$(date +"%Y-%m-%d %H:%M:%S")" > ${filename}.txt
echo Starts at "$(date +"%Y-%m-%d %H:%M:%S")" > ${filename}.csv
echo n_node, n_iter, n_node_param, n_iter_param, fallback_period, ad_scheme, hop_avg, hop_std, ttime, step_avg, step_std, port_avg, port_std >> ${filename}.csv


parallel -j${CORE_COUNT} run ::: "${nodes[@]}" ::: "${degree[0]}" ::: "${iter[0]}" ::: "${node_param[@]}" ::: "${iter_param[@]}" ::: "${periods[@]}" ::: "${dataset[0]}" ::: "${adschemes[@]}" ::: "${k[0]}" ::: "${filename}" ::: "${CORE_COUNT}"


echo Ends at "$(date +"%Y-%m-%d %H:%M:%S")" >> "${filename}".txt
echo Ends at "$(date +"%Y-%m-%d %H:%M:%S")" >> "${filename}".csv
