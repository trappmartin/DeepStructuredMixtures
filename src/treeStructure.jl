function buildTree(X::AbstractMatrix, y::AbstractVector, config::SPNGPConfig)

    N,D = size(X)
    @assert N == length(y)

    lowerBound = ones(D) * -Inf
    upperBound = ones(D) * Inf

    observations = collect(1:N)

    if config.sumRoot
        _buildSum(X, y, lowerBound, upperBound, config, 0, observations)
    else
        _buildSplit(X, y, lowerBound, upperBound, config, 0, observations)
    end
end

function _buildSplit(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::SPNGPConfig,
                    depth::Int,
                    observations::Vector{Int};
                    d = 1
                  )

    s = -Inf
    if size(X,1) > config.minData
        s_new = mean(X[:,d])

        z1 = 0
        z2 = 0

        while ((z1 == 0) || (z2 == 0))

            l = max(lowerBound[d], minimum(X[:,d]))
            u = min(upperBound[d], maximum(X[:,d]))
            v = u-l

            a = rand(Beta(2, 2))*v + l
            m = mean(X[:,d])

            s_new = (a + m) / 2

            z1 = sum(X[:,d] .<= s_new)
            z2 = sum(X[:,d] .> s_new)
        end
        s = s_new
    end

    split = Vector{Tuple{Int, Float64}}()
    if isfinite(s)
        push!(split, (d, s))
    end
    push!(split, (d, upperBound[d]))

    node = GPSplitNode(gensym("split"), Vector{Node}(), Vector{SPNNode}(),
                        lowerBound, upperBound, split)

    if isfinite(s)
        # first child
        lb = deepcopy(lowerBound)
        ub = deepcopy(upperBound)
        ub[d] = s
        idx = findall(X[:,d] .<= s)
        if (length(idx) > config.minData) && (depth < config.depth)
            if config.sumRoot
                dnew = d+1 > size(X,2) ? 1 : d+1
                add!(node,
                    _buildSum(X[idx,:], y[idx], lb, ub, config, depth, observations[idx], d = dnew)
                    )
            else
                add!(node,
                    _buildSplit(X[idx,:], y[idx], lb, ub, config, depth, observations[idx])
                    )
            end
        else
            add!(node, _buildGP(X[idx,:], y[idx], lb, ub, config, observations[idx]))
        end

        # second child
        lb = deepcopy(lowerBound)
        ub = deepcopy(upperBound)
        lb[d] = s
        idx = findall(X[:,d] .> s)
        if (length(idx) > config.minData) && (depth < config.depth)
            if config.sumRoot
                dnew = d+1 > size(X,2) ? 1 : d+1
                add!(node,
                    _buildSum(X[idx,:], y[idx], lb, ub, config, depth, observations[idx], d = dnew)
                    )
            else
                add!(
                    node, _buildSplit(X[idx,:], y[idx], lb, ub, config, observations[idx], depth)
                    )
            end
        else
            add!(node,
                _buildGP(view(X, idx, :), y[idx], lb, ub, config, observations[idx])
                )
        end
    else
        add!(node, _buildGP(X, y, lowerBound, upperBound, config, observations))
    end

    return node
end

function _buildSum(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::SPNGPConfig,
                    depth::Int,
                    observations::Vector{Int};
                    d = 1
                  )
    V = config.V
    w = rand(Dirichlet(V, 1.0))
    node = GPSumNode{Float64}(gensym("sum"), Vector{Node}(), Vector{SPNNode}(), Vector{Float64}())

    dims = if V > size(X,2)
        repeat(1:size(X,2), Int(ceil(V/size(X,2))))
    else
        collect(1:size(X,2))
    end

    for v = 1:V
        add!(node,
            _buildSplit(X, y, lowerBound, upperBound, config, depth+1, observations, d = dims[v]), log(w[v])
            )
    end
    return node
end

function _buildGP(X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::SPNGPConfig,
                    observations::Vector{Int} )

    if config.kernels isa AbstractVector

        w = rand(Dirichlet(length(config.kernels), 1.0))
        node = GPSumNode{Float64}(gensym("sum"), Vector{Node}(), Vector{SPNNode}(), Vector{Float64}())

        for v in 1:length(config.kernels)
            kern = deepcopy(config.kernels[v])
            obsNoise = copy(config.observationNoise)

            # create a full GP
            #gp = GP(X', y, MeanConst(mean(y)), kern, obsNoise)
            gp = GPE(; mean = MeanConst(mean(y)), kernel = kern, logNoise = obsNoise)
            gp.x = X'
            gp.y = y
            gp.nobs = length(y)
            gp.dim = size(X,2)
            gp.data = GaussianProcesses.KernelData(gp.kernel, gp.x, gp.x, gp.covstrat)
            gp.cK = GaussianProcesses.alloc_cK(gp.covstrat, gp.nobs)
            add!(node, GPNode(gensym("GP"), Vector{Node}(), gp, observations, collect(1:length(y))), log(w[v]))
        end
        return node
    else
        kern = deepcopy(config.kernels)
        obsNoise = copy(config.observationNoise)

        # create a full GP
        #gp = GP(X', y, MeanConst(mean(y)), kern, obsNoise)
        gp = GPE(; mean = MeanConst(mean(y)), kernel = kern, logNoise = obsNoise)
        gp.x = X'
        gp.y = y
        gp.nobs = length(y)
        gp.dim = size(X,2)
        gp.data = GaussianProcesses.KernelData(gp.kernel, gp.x, gp.x, gp.covstrat)
        gp.cK = GaussianProcesses.alloc_cK(gp.covstrat, gp.nobs)
        return GPNode(gensym("GP"), Vector{Node}(), gp, observations, collect(1:length(y)))
    end
end
