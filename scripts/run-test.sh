#!/bin/bash

export CORE_COUNT=8

declare -a nodes=(28 30 35 40 45 50)
declare degree=(4)
declare iter=(14)
declare k=(3)
declare -a node_param=(10 20 30)
declare -a iter_param=(10 20 30)
declare -a periods=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
declare dataset=("scratch")
declare -a adschemes=("replace" "add")

export CURRENT_DIR=$PWD
export RESULT_DIR=${CURRENT_DIR}/../results

export HOP_AVG_REGEX="\[Average Hop\] ([0-9\.e\-]+)"
export HOP_STD_REGEX="\[Standard Deviation Hop\]\ ([0-9\.e\-]+)"
export TEST_TIME_REGEX="\[Average Test Time\] ([0-9\.e\-]+) s"

run() {
    n_node=$1
    n_degree=$2
    n_iter=$3
    n_node_param=$4
    n_iter_param=$5
    period=$6
    dataset=$7
    ad_scheme=$8
    k=$9
    filename=${10}
    if [[ ${ad_scheme} == "add" ]]
    then
        file_solution="../poly_log/log${n_node_param}_${n_degree}_${k}_${n_iter_param}_same.pkl"
    else
        file_solution="../poly_log/log${n_node_param}_${n_degree}_${k}_${n_iter_param}_same_repl.pkl"
    fi
    
    if [[ ! -f "${file_solution}" ]]
    then
        return 1
    fi
    
    output=$(python3 safehiertopo.py -n ${n_node} -d ${n_degree} -i ${n_iter} -np ${n_node_param} -ip ${n_iter_param} -p ${period} -k ${k} -ds ${dataset} -a ${ad_scheme})
    
    echo $output >> ${filename}.txt
    
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
    
    echo ${n_node}, ${n_iter}, ${n_node_param}, ${n_iter_param}, ${period}, ${ad_scheme}, ${hop_avg}, ${hop_std}, ${ttime} >> ${filename}.csv
}

export -f run

filename="${RESULT_DIR}/fallback_period"

echo Starts at $(date +"%Y-%m-%d %H:%M:%S") > ${filename}.txt
echo Starts at $(date +"%Y-%m-%d %H:%M:%S") > ${filename}.csv
echo n_node, n_iter, n_node_param, n_iter_param, fallback_period, ad_scheme, hop_avg, hop_std, ttime >> ${filename}.csv


parallel -j${CORE_COUNT} run ::: ${nodes[@]} ::: ${degree} ::: ${iter} ::: ${node_param[@]} ::: ${iter_param[@]} ::: ${periods[@]} ::: ${dataset} ::: ${adschemes[@]} ::: ${k} ::: ${filename}


echo Ends at $(date +"%Y-%m-%d %H:%M:%S") >> ${filename}.txt
echo Ends at $(date +"%Y-%m-%d %H:%M:%S") >> ${filename}.csv
