module DeepGaussianProcessExperts

    using Reexport
    @reexport using SumProductNetworks
    using GaussianProcesses
    using RecipesBase
    using Distributions
    using StatsFuns

    import Base.rand
    import GaussianProcesses.predict
    import GaussianProcesses.optimize!

    export GPSumNode, GPSplitNode, GPNode, SPNGPConfig
    export getchild, leftGP, rightGP, predict, getx, gety, rand,
            buildTree, optimize!, stats, resample!, optim!, target

    # Type definitions
    struct GPSumNode{T<:Real} <: SumNode
        id::Symbol
        parents::Vector{<:Node}
        children::Vector{<:SPNNode}
        logweights::Vector{T}
    end

    struct GPSplitNode <: ProductNode
        id::Symbol
        parents::Vector{<:Node}
        children::Vector{<:SPNNode}
        lowerBound::Vector{Float64}
        upperBound::Vector{Float64}
        split::Vector{Tuple{Int, Float64}}
    end

    mutable struct GPNode <: Leaf
        id::Symbol
        parents::Vector{<:Node}
        dist::GaussianProcesses.GPBase
    end

    struct SPNGPConfig
        meanFunction
        kernels
        observationNoise::Float64
        minData::Int
        K::Int # number of splits per GPSplitNode
        V::Int # number of children under GPSumNode
        depth::Int # maximum depth (consecutive sum-product nodes)
        bnoise::Float64 # split noise
        sumRoot::Bool # use sum root
    end

    include("common.jl")
    include("treeStructure.jl")
    include("plot.jl")
    include("optimizeStructure.jl")

end
