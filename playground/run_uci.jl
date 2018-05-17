using GaussianProcesses, PGFPlots, SumProductNetworks
using StatsFuns, Distributions, JLD, ProgressMeter, MultivariateStats, FileIO
import SumProductNetworks.add!

using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "dataset"
            help = "dataset name"
            required = true
        "seed"
            arg_type = Int
            required = true
        "--splits"
            default = 4
            arg_type = Int
        "--overlap"
            default = 0.
            arg_type = Float64
    end

    return parse_args(s)
end

include("utilFunctions.jl")
include("dataTypes.jl")
include("dataTypeUtilFunctions.jl")
include("computationFunctions.jl")
include("regionGraph.jl")
include("regionGraphUtils.jl")
include("gpUtils.jl")


function main()
    parsed_args = parse_commandline()
    
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end

    dataset = parsed_args["dataset"]

    println("Processing dataset: $(dataset)")

    # process data sets
    f = File(format"JLD","../data/uci/$(dataset)")
    data = JLD.load(f);

    X = Float64.(data["X"])
    y = Float64.(data["y"]);

    N = size(X, 1)
    Int(ceil(N * 0.8)) # 80 / 20 split

    run = parsed_args["seed"]

    srand(run)

    Ntrain = Int(ceil(N * 0.8))
    Dy = size(y, 2)

    ids = collect(1:size(X, 1))
    trainingIds = shuffle(ids)[1:Ntrain]
    testingIds = shuffle(setdiff(ids, trainingIds));

    Xtest = X[testingIds,:]
    ytest = y[testingIds, :]

    Xtrain = X[trainingIds, :]
    ytrain = y[trainingIds, :];

    # mean prediction
    meanY = mean(ytrain,1)
    rmse_mean = sqrt(mean((ytest .- meanY).^2))

    # LLS prediction
    a = llsq(Xtrain, ytrain, bias = false)

    yhat = Xtest * a
    rmse_lls = sqrt(mean((yhat .- ytest).^2))

    # ridge regression prediction
    α = 0.01
    a = ridge(Xtrain, ytrain, α, bias = false)

    yhat = Xtest * a
    rmse_ridge = sqrt(mean((yhat .- ytest).^2))

    # random forest regression


    # gaussian process regression (if N < 1000)
    rmse_gp_fixed_noise = Inf
    rmse_gp_opt_noise = Inf

    if Ntrain < 1000
        mZero = MeanZero()
        kern = SE(-1.0,0.0) 
        logObsNoise = -1.0

        yhat_fixed = zeros(size(ytest))
        yhat_opt = zeros(size(ytest))

        for yi in 1:Dy

            # fixed noise
            gp_ = GP(Xtrain', vec(ytrain[:,yi]), MeanZero(), SE(-1.0,0.0), -1.)
            #optimize2!(gp_, mean = false, kern = true, noise = false, lik=false)

            μ, σ² = predict_y(gp_, Xtest')
            yhat_fixed[:,yi] = μ

            # optimized noise
            #gp_ = GP(Xtrain', vec(ytrain[:,yi]), MeanZero(), SE(-1.0,0.0), -1.)
            #optimize2!(gp_, mean = false, kern = true, noise = true, lik=false)

            #μ, σ² = predict_y(gp_, Xtest');
            #yhat_opt[:,yi] = μ
        end

        rmse_gp_fixed_noise = sqrt(mean((yhat_fixed .- ytest).^2))
        #rmse_gp_opt_noise = sqrt(mean((yhat_opt .- ytest).^2))
    end

    # SPN-GP
    (N, D) = size(Xtrain)

    global gID = 1

    numSums = 1
    meanFunction = MeanZero();
    kernelFunctions = [LinIso(log(5.0)), SE(-1., 0.), Mat32Iso(log(5.0), log(1.0))]

    kernelPriors = []

    noise = -1.;

    # data range
    minX = vec(minimum(X, 1)) - 0.1
    maxX = vec(maximum(X, 1)) + 0.1

    # split size
    δ = (maxX - minX) ./ parsed_args["splits"]

    # maximum depth
    max_depth = 1
    min_samples = 500

    overlap = parsed_args["overlap"]

    (rootRegion, sumRegions, gpRegions, allPartitions) = poonDomingos_ND(δ, minX, maxX, max_depth, min_samples, Xtrain);

    RegionIDs = Dict(r[2] => r[1] for r in enumerate(union(sumRegions, gpRegions)));
    PartitionIDS = Dict(p[2] => p[1] + maximum(values(RegionIDs)) for p in enumerate(allPartitions));

    yhat_fixed = zeros(size(ytest))
    yhat_opt = zeros(size(ytest))

    for yi in 1:Dy

        global gID = 1

        # optimize noise
        root_ = convertToSPN_ND(rootRegion, gpRegions, RegionIDs, PartitionIDS, Xtrain, ytrain[:,yi], meanFunction, 
            kernelFunctions, kernelPriors, noise; overlap = overlap, do_mcmc = false)

        gpnodes = unique(filter(n -> isa(n, GPLeaf), SumProductNetworks.getOrderedNodes(root_)));
        #map(gnode -> optimize2!(gnode.gp, mean = false, kern = true, noise = true, lik=false), gpnodes);

        fill!(root_.prior_weights, 1. / length(root_))
        fill!(root_.posterior_weights, 1. / length(root_))

        spn_update!(root_)
        spn_posterior(root_)

        μ = predict_spn!(root_, Xtest);
        yhat_fixed[:,yi] = μ

        #global gID = 1

        # fixed noise
        #root_ = convertToSPN_ND(rootRegion, gpRegions, RegionIDs, PartitionIDS, Xtrain, ytrain[:,yi], meanFunction, 
        #    kernelFunctions, kernelPriors, noise; overlap = overlap, do_mcmc = false)

        #gpnodes = unique(filter(n -> isa(n, GPLeaf), SumProductNetworks.getOrderedNodes(root_)));
        #map(gnode -> optimize2!(gnode.gp, mean = false, kern = true, noise = false, lik=false), gpnodes);

        #fill!(root_.prior_weights, 1. / length(root_))
        #fill!(root_.posterior_weights, 1. / length(root_))

        #spn_update!(root_)
        #spn_posterior(root_)

        #μ = predict_spn!(root_, Xtest);
        #yhat_opt[:,yi] = μ
    end

    rmse_spn_fixed_noise = sqrt(mean((yhat_fixed .- ytest).^2))
    #rmse_spn_opt_noise = sqrt(mean((yhat_opt .- ytest).^2))

    #writecsv("$(dataset)_$(run).csv", [rmse_mean, rmse_lls, rmse_ridge, rmse_gp_fixed_noise, rmse_gp_opt_noise,
    #        rmse_spn_fixed_noise, rmse_spn_opt_noise])

    writecsv("$(dataset)_$(run).csv", [rmse_mean, rmse_lls, rmse_ridge, rmse_gp_fixed_noise, rmse_spn_fixed_noise])

    info("Finished run: $(run) on dataset: $(dataset)")
end

main()