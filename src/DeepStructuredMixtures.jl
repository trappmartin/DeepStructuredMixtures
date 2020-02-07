module DeepStructuredMixtures

    using Reexport
    @reexport using SumProductNetworks
    using RecipesBase
    using Distributions
    using StatsFuns
    using LinearAlgebra
    using AxisArrays

    import Base.rand
    import Base.get

    import SumProductNetworks.scope
    import SumProductNetworks.hasscope
    import SumProductNetworks.hasobs
    import SumProductNetworks.params

    import StatsBase.params

    export GPSumNode, GPSplitNode, GPNode, DSMGPConfig
    export DSMGP, PoE, gPoE, rBCM
    export getchild, leftGP, rightGP, predict, getx, gety, rand,
            buildTree, optimize!, stats, resample!, optim!, target

    const ϵ = 1e-8

    include("AdvancedCholeskey.jl")

    # custom block-diagonal matrix type
    include("spnmatrix.jl")

    # GP related codes
    include("means.jl")
    include("kernels.jl")
    include("gaussianprocess.jl")

    # Type definitions
    struct GPSumNode{T<:Real,C<:SPNNode} <: SumNode{T}
        id::Symbol
        parents::Vector{<:Node}
        children::Vector{C}
        logweights::Vector{T}
    end

    function Base.show(io::IO, ::MIME"text/plain", m::GPSumNode)
        print(io, "GP Sum Node [$(m.id)] \n weights: ",exp.(m.logweights))
    end
    Base.show(io::IO, m::GPSumNode) = print(io, "GPSumNode(",m.id,")")

    struct GPSplitNode <: ProductNode
        id::Symbol
        parents::Vector{<:Node}
        children::Vector{<:SPNNode}
        lowerBound::Vector{Float64}
        upperBound::Vector{Float64}
        split::Vector{Tuple{Int, Float64}}
    end

    struct GPNode <: Leaf
        id::Symbol
        parents::Vector{<:Node}
        dist::GaussianProcess
        observations::BitArray{1}
        obs::Vector{Int}
        lb::Vector{Float64}
        ub::Vector{Float64}
        nobs::Int
        kernelid::Int
    end


    @inline id(node::Node) = node.id
    @inline id(node::GPNode) = node.id

    params(node::GPNode) = params(node.dist)

    @inline hasscope(node::GPNode) = true
    @inline hasscope(node::GPSumNode) = true
    @inline hasscope(node::GPSplitNode) = true

    @inline scope(node::GPNode) = node.observations
    @inline scope(node::GPSplitNode) = mapreduce(scope, vcat, children(node))
    @inline scope(node::GPSumNode) = scope(node[1])

    @inline hasobs(node::GPNode) = false
    @inline hasobs(node::GPSplitNode) = false
    @inline hasobs(node::GPSumNode) = false

    struct DSMGPConfig
        meanFun::Union{Nothing,MeanFunction}
        kernels::Union{KernelFunction, Vector{KernelFunction}}
        observationNoise::Float64
        minData::Int
        K::Int # number of splits per GPSplitNode
        V::Int # number of children under GPSumNode
        depth::Int # maximum depth (consecutive sum-product nodes)
        bnoise::Float64 # split noise
        sumRoot::Bool # use sum root
    end

    struct BiDict
        x::Dict
        fx::Dict
    end

    struct DSMGP{T<:Real}
        root::Node
        D::Matrix{T}
        gpmap::BiDict
    end

    struct PoE{T<:Real}
        root::GPSplitNode
        D::Matrix{T}
        gpmap::BiDict
    end

    struct gPoE{T<:Real}
        root::GPSplitNode
        D::Matrix{T}
        gpmap::BiDict
    end

    struct rBCM{T<:Real}
        root::GPSplitNode
        D::Matrix{T}
        gpmap::BiDict
    end

    # codes
    include("common.jl")
    include("treeStructure.jl")
    include("fit.jl")
    include("optimize.jl")
    include("optimisers.jl")
    include("finetuning.jl")

    # utilities
    include("plot.jl")
    include("datasets.jl")
    include("scorefunctions.jl")
end
