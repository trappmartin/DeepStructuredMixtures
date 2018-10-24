DATE=`date +%Y-%m-%d-%H`
mkdir $DATE

for dataset in concrete energy ccpp
do
    for run in 1 2 3 4 5
    do
        ~/julia/bin/julia --color=yes run_uci.jl --splits 4 --overlap 0.2 $dataset $DATE $run
    done
done