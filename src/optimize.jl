using Base.Threads
using Optim

export getparams, set_params!
export optimize_restarts!, rand_init!
export ∇mll, mll, mll!

function rand_init!(spn::Union{GPSplitNode, GPSumNode}, D)
    gp = leftGP(spn)
    n = gp isa Array ? sum(map(sum, nparams.(gp))) : sum(nparams(gp))
    hyp = randn(n)

    setparams!(spn, hyp)
    fit!(spn, D)
    return spn
end

@inline mll(node::GPNode) = mll(node.dist)
@inline mll(node::GPSplitNode) = mapreduce(mll, +, children(node))
function mll(node::GPSumNode)
    K = length(node)
    StatsFuns.logsumexp(map(c -> -log(K)+mll(c), children(node)))
end

@inline mll(model::Union{DSMGP,PoE}) = mll(model.root)

function mll!(node::GPNode, ℓ::AxisArray)
    ℓ[node.id] = mll(node.dist)
    return ℓ[node.id]
end
function mll!(node::GPSplitNode, ℓ::AxisArray)
    ℓ[node.id] = mapreduce(c -> mll!(c, ℓ), +, children(node))
    return ℓ[node.id]
end
function mll!(node::GPSumNode, ℓ::AxisArray)
    K = length(node)
    ℓ[node.id] = StatsFuns.logsumexp(map(c -> -log(K)+mll!(c, ℓ), children(node)))
    return ℓ[node.id]
end

# == gradient propagation (global) ==
@inline function ∇mll!(node::GPNode,
                       ∇parent::Float64,
                       lρ::Float64,
                       ℓ::AxisArray,
                       logS::Float64,
                       ∇::AbstractVector)
    w = exp(-logS+lρ+ℓ[node.id]+∇parent)
    ∇[:] += ∇mll(node.dist)*w
end

function ∇mll!(node::GPSplitNode,
               ∇parent::Float64,
               lρ::Float64,
               ℓ::AxisArray,
               logS::Float64,
               ∇::AbstractVector)
    Threads.@threads for child = children(node)
        lp = ℓ[node.id] - ℓ[child.id]
        ∇mll!(child, ∇parent + lp, lρ, ℓ, logS, ∇)
    end
end

function ∇mll!(node::GPSumNode,
               ∇parent::Float64,
               lρ::Float64,
               ℓ::AxisArray,
               logS::Float64,
               ∇::AbstractVector)
    K = length(node)
    @inbounds for child in children(node)
        ∇mll!(child, -log(K)+∇parent, log(K)+lρ, ℓ, logS, ∇)
    end
end

function ∇mll!(node::GPSumNode{T,V},
               ∇parent::Float64,
               lρ::Float64,
               ℓ::AxisArray,
               logS::Float64,
               ∇::AbstractVector) where {T<:AbstractFloat, V<:GPNode}
    K = length(node)
    c = 1
    @inbounds for k = 1:K
        n = sum(nparams(node[k].dist))-1
        ∇mll!(node[k], ∇parent, lρ, ℓ, logS, view(∇, c:(c+n)) )
        c += n+1
    end
end

# == gradient propagation (finetune) ==
@inline function ∇mll!(node::GPNode,
                       ∇parent::Float64,
                       lρ::Float64,
                       ℓ::AxisArray,
                       logS::Float64,
                       ∇::AbstractVector,
                       D::AbstractVector,
                       gpmap::BiDict)
    w = exp(-logS+lρ+ℓ[node.id]+∇parent)
    ∇[:] += ∇mll(node.dist)*w*D[gpmap.x[node.id]]
end

function ∇mll!(node::GPSplitNode,
               ∇parent::Float64,
               lρ::Float64,
               ℓ::AxisArray,
               logS::Float64,
               ∇::AbstractVector,
               D::AbstractVector,
               gpmap::BiDict)
    K = length(node)

    Threads.@threads for k = 1:K
        lp = ℓ[node.id] - ℓ[node[k].id]
        ∇mll!(node[k], ∇parent + lp, lρ, ℓ, logS, ∇, D, gpmap)
    end
end

function ∇mll!(node::GPSumNode,
               ∇parent::Float64,
               lρ::Float64,
               ℓ::AxisArray,
               logS::Float64,
               ∇::AbstractVector,
               D::AbstractVector,
               gpmap::BiDict)
    K = length(node)
    @inbounds for child in children(node)
        ∇mll!(child, -log(K)+∇parent, log(K)+lρ, ℓ, logS, ∇, D, gpmap)
    end
end

function ∇mll!(node::GPSumNode{T,V},
               ∇parent::Float64,
               lρ::Float64,
               ℓ::AxisArray,
               logS::Float64,
               ∇::AbstractVector,
               D::AbstractVector,
               gpmap::BiDict) where {T<:AbstractFloat, V<:GPNode}
    @warn "should not be here.."
    K = length(node)
    c = 1
    @inbounds for k = 1:K
        n = sum(nparams(node[k].dist))-1
        ∇mll!(node[k], ∇parent, lρ, ℓ, logS, view(∇, c:(c+n)), D, gpmap)
        c += n+1
    end
end


function ∇mll(node::GPNode, ∇parent::Float64, ℓ, logS, ::Val{true}; kwargs...)
    ∇ = mll(node.dist)
    w = ℓ[node.id] -logS + ∇parent
    ∇ *= exp(w)
    return Dict(node.id => ∇)
end

function ∇mll(node::GPSplitNode, ∇parent::Float64, ℓ, logS, soft::Val{true}; kwargs...)
    K = length(node)

    lpchildren = map(c -> ℓ[c.id], children(node))

    ∇ = ∇mll(node[1], ∇parent + sum(lpchildren[2:end]), ℓ, logS, soft; kwargs...)
    @inbounds for k = 2:K
        merge!(∇, ∇mll(node[k], ∇parent + sum(lpchildren[vcat(1:k-1, k+1:end)]), ℓ, logS, soft; kwargs...))
    end
    return ∇
end

function ∇mll(node::GPSumNode, ∇parent::Float64, ℓ, logS, soft::Val{true}; kwargs...)
    K = length(node)
    return mapreduce(c -> ∇mll(c, ∇parent - log(K), ℓ, logS, soft; kwargs...), merge, children(node))
end

function ∇mll(node::SPNNode, soft; kwargs...)
    nodes = SumProductNetworks.getOrderedNodes(node)
    ids = map(n -> n.id, nodes)
    L = AxisArray(zeros(length(ids)), ids)
    mll!(node, L)
    ∇mll(node, 0.0, L, L[node.id], soft; kwargs...)
end

getparams(node::GPNode) = Dict(node.id => vcat(DeepStructuredMixtures.params(node.dist)...))
getparams(node::Node) = mapreduce(c -> getparams(c), merge, children(node))

setparams!(node::GPNode, params) = setparams!(node.dist, params)
setparams!(node::SPNNode, params) = setparams!(node, params)
function setparams!(node::GPSumNode{T,V}, params) where {T<:AbstractFloat,V<:GPNode}
    c = 1
    for child in children(node)
        n = sum(nparams(child.dist))-1
        setparams!(child.dist, params[c:(c+n)])
        c += n+1
    end
end
setparams!(node::Node, params) = map(c-> setparams!(c, params), children(node))

@inline getLeafIds(node::GPNode) = node.id
@inline getLeafIds(node::Node) = mapreduce(getLeafIds, vcat, children(node))

