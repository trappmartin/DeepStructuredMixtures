module DeepGaussianProcessExperts

    using Reexport
    @reexport using SumProductNetworks
    using GaussianProcesses
    using RecipesBase
    using Distributions
    using StatsFuns
    using PDMats: PDMat
    using LinearAlgebra, SparseArrays
    using AxisArrays

    import Base.rand
    import GaussianProcesses.predict
    import GaussianProcesses.optimize!
    import GaussianProcesses.update_mll!

    import SumProductNetworks.scope
    import SumProductNetworks.hasscope
    import SumProductNetworks.hasobs
    import SumProductNetworks.params

    export GPSumNode, GPSplitNode, GPNode, SPNGPConfig
    export getchild, leftGP, rightGP, predict, getx, gety, rand,
            buildTree, optimize!, stats, resample!, optim!, target

    # Type definitions
    struct GPSumNode{T<:Real} <: SumNode{T}
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
        observations::BitArray{1}
        firstobs::Int
        lastobs::Int
        nobs::Int
    end

    params(node::GPNode) = GaussianProcesses.get_params(node.dist)

    @inline hasscope(node::GPNode) = true
    @inline hasscope(node::GPSumNode) = true
    @inline hasscope(node::GPSplitNode) = true

    @inline scope(node::GPNode) = node.observations
    @inline scope(node::GPSplitNode) = mapreduce(scope, vcat, children(node))
    @inline scope(node::GPSumNode) = scope(node[1])

    @inline hasobs(node::GPNode) = false
    @inline hasobs(node::GPSplitNode) = false
    @inline hasobs(node::GPSumNode) = false

    struct SPNGPConfig
        kernels
        observationNoise::Float64
        minData::Int
        K::Int # number of splits per GPSplitNode
        V::Int # number of children under GPSumNode
        depth::Int # maximum depth (consecutive sum-product nodes)
        bnoise::Float64 # split noise
        sumRoot::Bool # use sum root
    end

    include("AdvancedCholeskey.jl")

    include("common.jl")
    include("treeStructure.jl")
    include("plot.jl")
    include("optimizeStructure.jl")
    include("fit.jl")

end
