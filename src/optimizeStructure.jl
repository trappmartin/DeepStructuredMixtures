function resample!(node::GPSplitNode, X::AbstractArray,
    y::AbstractVector, λ::Float64, dict::Dict{Symbol,Float64})
    if !isempty(node.split)
        d, s_old = first(node.split)
        l = max(node.lowerBound[d], minimum(X[:,d]))
        u = min(node.upperBound[d], maximum(X[:,d]))
        v = u-l

        a = rand(Beta(2, 2))*v + l

        s_new = (λ * a + (1-λ) * s_old)
        dict[node.id] = s_new

        idx = findall(X[:,d] .<= s_new)
        lp = length(idx) > 1 ? resample!(node[1], X[idx,:], y[idx], λ, dict) : -Inf
        idx = findall(X[:,d] .> s_new)
        lp += length(idx) > 1 ? resample!(node[2], X[idx,:], y[idx], λ, dict) : -Inf
        return lp
    else
        return resample!(node[1], X, y, λ, dict = dict)
    end
end

function resample!(node::GPSumNode, X::AbstractArray,
    y::AbstractVector, λ::Float64, dict::Dict{Symbol,Float64})

    lp = map(c -> resample!(c, X, y, λ, dict), children(node))
    return StatsFuns.logsumexp(lp + logweights(node))
end

function resample!(node::GPNode, X::AbstractArray,
    y::AbstractVector, λ::Float64, dict::Dict{Symbol,Float64})

    meanFun = deepcopy(node.dist.mean)
    kern = deepcopy(node.dist.kernel)
    obsNoise = copy(node.dist.logNoise)

    # create a full GP
    gp = GP(X', y, meanFun, kern, obsNoise)
    GaussianProcesses.update_target!(gp)
    return gp.target
end

function updateStructure!(node::GPSplitNode, X::AbstractArray,
    y::AbstractVector, dict::Dict{Symbol,Float64})
    if !isempty(node.split)
        d, s_old = node.split[1]
        s_new = dict[node.id]
        node.split[1] = (d, s_new)
        if isfinite(node.split[2][2])
            node.split[2] = (d, maximum(X[:,d]))
        end

        idx = findall(X[:,d] .<= s_new)
        updateStructure!(node[1], X[idx,:], y[idx], dict)
        idx = findall(X[:,d] .> s_new)
        updateStructure!(node[2], X[idx,:], y[idx], dict)
    else
        updateStructure!(node[1], X, y, dict)
    end
end

function updateStructure!(node::GPSumNode, X::AbstractArray,
    y::AbstractVector, dict::Dict{Symbol,Float64})
    for child in children(node)
        updateStructure!(child, X, y, dict)
    end
end

function updateStructure!(node::GPNode, X::AbstractArray,
    y::AbstractVector, dict::Dict{Symbol,Float64})

    meanFun = deepcopy(node.dist.mean)
    kern = deepcopy(node.dist.kernel)
    obsNoise = copy(node.dist.logNoise)

    # create a full GP
    node.dist = GP(X', y, meanFun, kern, obsNoise)
    GaussianProcesses.update_target!(node.dist)
end

function target(node::GPSplitNode)
    return sum(target, children(node))
end

function target(node::GPSumNode)
    lp = map(c -> target(c), children(node))
    return StatsFuns.logsumexp(lp + logweights(node))
end

function target(node::GPNode)
    return node.dist.target
end
