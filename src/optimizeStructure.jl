function resample!(node::GPSplitNode, X::AbstractArray,
    y::AbstractVector, λ::Float64, maxdata::Int)

    dict = Dict{Symbol,Float64}()
    dict[node.id] = -Inf

    lp = 0.0

    if length(node.split) > 1
        @assert length(node) == 2
        d, s_old = first(node.split)
        l = max(node.lowerBound[d], minimum(X[:,d]))
        u = min(node.upperBound[d], maximum(X[:,d]))
        v = u-l

        a = rand(Beta(2, 2))*v + l

        s_new = (λ * a + (1-λ) * s_old)
        dict[node.id] = s_new

        idx = findall(X[:,d] .<= s_new)
        lp += length(idx) > maxdata ? -Inf : 0
        if length(idx) > 1
            lp_, dict_ = resample!(node[1], X[idx,:], y[idx], λ, maxdata)
            lp += lp_
            merge!(dict, dict_)
        else
            lp += -Inf
        end

        idx = findall(X[:,d] .> s_new)
        lp += length(idx) > maxdata ? -Inf : 0
        if length(idx) > 1
            lp_, dict_ = resample!(node[2], X[idx,:], y[idx], λ, maxdata)
            lp += lp_
            merge!(dict, dict_)
        else
            lp += -Inf
        end
    else
        @assert length(node) == 1
        lp_, dict_ = resample!(node[1], X, y, λ, maxdata)
        lp += lp
        merge!(dict, dict_)
    end
    return lp, dict
end

function resample!(node::GPSumNode, X::AbstractArray,
    y::AbstractVector, λ::Float64, maxdata::Int)

    dict = Dict{Symbol,Float64}()
    lp = zeros(length(node))
    for (k, child) in enumerate(children(node))
        lp_, dict_ = resample!(child, X, y, λ, maxdata)
        lp[k] = lp_
        merge!(dict, dict_)
    end
    return StatsFuns.logsumexp(lp + logweights(node)), dict
end

function resample!(node::GPNode, X::AbstractArray,
    y::AbstractVector, λ::Float64, maxdata::Int)

    meanFun = deepcopy(node.dist.mean)
    kern = deepcopy(node.dist.kernel)
    obsNoise = copy(node.dist.logNoise)

    # create a full GP
    gp = GP(X', y, meanFun, kern, obsNoise)
    GaussianProcesses.update_target!(gp)
    return gp.target, Dict{Symbol,Float64}()
end

function updateStructure!(node::GPSplitNode, X::AbstractArray,
    y::AbstractVector, dict::Dict{Symbol,Float64}, upperBound::Vector{Float64})
    if length(node.split) > 1
        d, s_old = node.split[1]
        s_new = get(dict, node.id, node.split[1][2])
        node.split[1] = (d, s_new)
        if isfinite(node.split[2][2])
            node.split[2] = (d, upperBound[d])
        end

        idx = findall(X[:,d] .<= s_new)
        ub = deepcopy(upperBound)
        ub[d] = s_new
        updateStructure!(node[1], X[idx,:], y[idx], dict, ub)
        idx = findall(X[:,d] .> s_new)
        updateStructure!(node[2], X[idx,:], y[idx], dict, upperBound)
    else
        updateStructure!(node[1], X, y, dict, upperBound)
    end
end

function updateStructure!(node::GPSumNode, X::AbstractArray,
    y::AbstractVector, dict::Dict{Symbol,Float64}, upperBound::Vector{Float64})
    for child in children(node)
        updateStructure!(child, X, y, dict, upperBound)
    end
end

function updateStructure!(node::GPNode, X::AbstractArray,
    y::AbstractVector, dict::Dict{Symbol,Float64}, upperBound::Vector{Float64})

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



function optim!(root, x, y, n; t0 = 0.1, tn = 50, λ0 = 0.1, λ1 = 0.6, maxdata = 100, m = 10)

    v = Vector{Float64}()
    push!(v, DeepGaussianProcessExperts.target(root))

    overallBest = Dict{Symbol,Float64}()
    bestScore = v[1]

    ids = stats(root)[:ids]

    for i in 1:n
        @info "Iteration: ", i
        bestConfig = Dict{Symbol,Float64}()
        score = -Inf
        for r in 1:m
            λ = rand()*(λ1-λ0) + λ0
            s, dict = DeepGaussianProcessExperts.resample!(root, x, y, λ, maxdata)

            if isfinite(s)
                if (s > score)
                    score = s
                    bestConfig = dict
                end
            end
        end

        t = t0 - i * (t0 - tn)/n

        if (exp(-(-score + bestScore)/t) > rand()) || score > bestScore
            ub = ones(size(x,2)) * Inf
            DeepGaussianProcessExperts.updateStructure!(root, x, y, bestConfig, ub)
            push!(v, score)
            bestScore = score
            if minimum(v) == score
                overallBest = bestConfig
            end
            @info "Accepted new solution"
        end
    end
    ub = ones(size(x,2)) * Inf
    DeepGaussianProcessExperts.updateStructure!(root, x, y, overallBest, ub)

    return (root, v)
end
