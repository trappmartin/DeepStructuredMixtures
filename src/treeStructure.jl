function buildTree(X::AbstractMatrix, y::AbstractVector, config::SPNGPConfig)

    N,D = size(X)
    @assert N == length(y)

    lowerBound = ones(D) * -Inf
    upperBound = ones(D) * Inf
    dims = collect(1:D)

    if D > 1
        _buildSum(X, y, lowerBound, upperBound, dims, config, 0)
    else
        _buildSplit(X, y, lowerBound, upperBound, dims, config, 0)
    end
end

function _buildSplit(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    dims::Vector{Int},
                    config::SPNGPConfig,
                    depth::Int;
                    d = 1
                  )

    s = if size(X,1) > config.minData
        l = max(lowerBound[d], minimum(X[:,d]))
        u = min(upperBound[d], maximum(X[:,d]))
        v = u-l

        a = rand(Beta(2, 2))*v + l
        m = mean(X[:,d])

        (a + m) / 2
    else
        -Inf
    end

    split = Vector{Tuple{Int, Float64}}()
    if isfinite(s)
        push!(split, (d, s))
        push!(split, (d, upperBound[d]))
    end

    node = GPSplitNode(gensym("split"), Vector{Node}(), Vector{SPNNode}(),
                        lowerBound, upperBound, split)

    # first child
    lb = deepcopy(lowerBound)
    ub = deepcopy(upperBound)
    ub[d] = s
    idx = findall(X[:,d] .<= s)
    if (length(idx) > config.minData) && (depth < config.depth)
        if size(X,2) > 1
            dnew = d+1 > size(X,2) ? 1 : d+1
            add!(node, _buildSum(view(X, idx, :), y[idx], lb, ub, dims, config, depth, d = dnew))
        else
            add!(node, _buildSplit(view(X, idx, :), y[idx], lb, ub, dims, config, depth))
        end
    else
        add!(node, _buildGP(view(X, idx, :), y[idx], lb, ub, dims, config))
    end

    # second child
    lb = deepcopy(lowerBound)
    ub = deepcopy(upperBound)
    lb[d] = s
    idx = findall(X[:,d] .> s)
    if (length(idx) > config.minData) && (depth < config.depth)
        if size(X,2) > 1
            dnew = d+1 > size(X,2) ? 1 : d+1
            add!(node, _buildSum(view(X, idx, :), y[idx], lb, ub, dims, config, depth, d = dnew))
        else
            add!(node, _buildSplit(view(X, idx, :), y[idx], lb, ub, dims, config, depth))
        end
    else
        add!(node, _buildGP(view(X, idx, :), y[idx], lb, ub, dims, config))
    end
    return node
end

function _buildSum(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    dims::Vector{Int},
                    config::SPNGPConfig,
                    depth::Int;
                    d = 1
                  )
    V = min(config.V, size(X,2))
    w = rand(Dirichlet(V, 1.0))
    node = GPSumNode{Float64}(gensym("sum"), Vector{Node}(), Vector{SPNNode}(), Vector{Float64}())

    for v = 1:V
        add!(node, _buildSplit(X, y, lowerBound, upperBound, dims, config, depth+1, d = v), log(w[v]))
    end
    return node
end

function _buildGP(X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    dims::Vector{Int},
                    config::SPNGPConfig )

    if config.kernels isa AbstractVector

        w = rand(Dirichlet(length(config.kernels), 1.0))
        node = GPSumNode{Float64}(gensym("sum"), Vector{Node}(), Vector{SPNNode}(), Vector{Float64}())

        for v in 1:length(config.kernels)
            meanFun = deepcopy(config.meanFunction)
            kern = deepcopy(config.kernels[v])
            obsNoise = copy(config.observationNoise)

            # create a full GP
            gp = GP(X', y, meanFun, kern, obsNoise)
            add!(node, GPNode(gensym("GP"), Vector{Node}(), gp), log(w[v]))
        end
        return node
    else
        meanFun = deepcopy(config.meanFunction)
        kern = deepcopy(config.kernels)
        obsNoise = copy(config.observationNoise)

        # create a full GP
        gp = GP(X', y, meanFun, kern, obsNoise)
        return GPNode(gensym("GP"), Vector{Node}(), gp)
    end
end
