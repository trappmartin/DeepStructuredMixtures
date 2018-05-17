for dataset in ccpp
do
    for run in 1 2 3 4 5
    do
        ~/julia/bin/julia --color=yes run_uci.jl --splits 4 --overlap 0.2 $dataset $run
    done
done