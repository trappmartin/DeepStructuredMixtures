function buildTree(X::AbstractMatrix, y::AbstractVector, config::SPNGPConfig)

    N,D = size(X)
    @assert N == length(y)

    lowerBound = ones(D) * -Inf
    upperBound = ones(D) * Inf

    if config.sumRoot
        _buildSum(X, y, lowerBound, upperBound, config, 0)
    else
        _buildSplit(X, y, lowerBound, upperBound, config, 0)
    end
end

function _buildSplit(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::SPNGPConfig,
                    depth::Int;
                    d = 1
                  )

    s = -Inf
    if size(X,1) > config.minData
        @info "here"
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
        @info "here 1"
        push!(split, (d, s))
    end
    push!(split, (d, upperBound[d]))

    node = GPSplitNode(gensym("split"), Vector{Node}(), Vector{SPNNode}(),
                        lowerBound, upperBound, split)

    if isfinite(s)

        @info "here 2"
        # first child
        lb = deepcopy(lowerBound)
        ub = deepcopy(upperBound)
        ub[d] = s
        idx = findall(X[:,d] .<= s)
        if (length(idx) > config.minData) && (depth < config.depth)
            if config.sumRoot
                dnew = d+1 > size(X,2) ? 1 : d+1
                add!(node, _buildSum(view(X, idx, :), y[idx], lb, ub, config, depth, d = dnew))
            else
                add!(node, _buildSplit(view(X, idx, :), y[idx], lb, ub, config, depth))
            end
        else
            add!(node, _buildGP(view(X, idx, :), y[idx], lb, ub, config))
        end

        # second child
        lb = deepcopy(lowerBound)
        ub = deepcopy(upperBound)
        lb[d] = s
        idx = findall(X[:,d] .> s)
        if (length(idx) > config.minData) && (depth < config.depth)
            if config.sumRoot
                dnew = d+1 > size(X,2) ? 1 : d+1
                add!(node, _buildSum(view(X, idx, :), y[idx], lb, ub, config, depth, d = dnew))
            else
                add!(node, _buildSplit(view(X, idx, :), y[idx], lb, ub, config, depth))
            end
        else
            add!(node, _buildGP(view(X, idx, :), y[idx], lb, ub, config))
        end
    else
        add!(node, _buildGP(X, y, lowerBound, upperBound, config))
    end

    return node
end

function _buildSum(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::SPNGPConfig,
                    depth::Int;
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
        add!(node, _buildSplit(X, y, lowerBound, upperBound, config, depth+1, d = dims[v]), log(w[v]))
    end
    return node
end

function _buildGP(X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
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
