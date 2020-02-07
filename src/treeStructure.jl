export build
export buildDSMGP, buildPoE, buildBCM

function buildTree(X::AbstractMatrix, y::AbstractVector, config::DSMGPConfig)

    N,D = size(X)
    @assert N == length(y)

    lowerBound = ones(D) * -Inf
    upperBound = ones(D) * Inf

    observations = collect(1:N)

    @assert all(isfinite.(X))

    if config.sumRoot
        _buildSum(X, y, lowerBound, upperBound, config, 0, observations, N)
    else
        _buildSplit(X, y, lowerBound, upperBound, config, 0, observations, N)
    end
end

function getSplits(X::AbstractMatrix{T},
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    minData::Int,
                    ϵ::Float64,
                    K::Int,
                    d::Int;
                    depth = 1) where {T<:Real}
    α = β = 2.0

    K_ = depth^2
    s = Vector{Float64}()

    l = max(lowerBound[d], minimum(X[:,d]))
    u = min(upperBound[d], maximum(X[:,d]))
    v = u-l

    idx = findall((X[:,d] .> l) .& (X[:,d] .<= u))
    if length(idx) > minData*2
        s_new = Float64(mean(X[:,d]))

        z1 = 0
        z2 = 0

        c = 0
        m = mean(X[idx,d])
        m = median(X[idx,d])

        while ((z1 == 0) || (z2 == 0))
            a = rand(Beta(α, β))*v + l

            s_new = Float64(ϵ*a + (1-ϵ)*m)

            z1 = sum(X[idx,d] .<= s_new)
            z2 = sum(X[idx,d] .> s_new)

            c += 1

            if c > 100
                @warn z1, z2, s_new, m, a
                return s
            end
        end

        zi = rand(1:2)
        if zi == 1
            if (z1 > minData) && (K_ < K)
                ub = copy(upperBound)
                ub[d] = s_new
                append!(s, getSplits(X,
                          lowerBound,
                          ub,
                          minData,
                          ϵ,
                          K,
                          d,
                          depth = depth+1
                         ))
                K_ += 1
            end
            if (z2 > minData) && (K_ < K)
                lb = copy(upperBound)
                lb[d] = s_new
                append!(s, getSplits(X,
                          lb,
                          upperBound,
                          minData,
                          ϵ,
                          K,
                          d,
                          depth = depth+1
                         ))
            end
        else
            if (z2 > minData) && (K_ < K)
                lb = copy(upperBound)
                lb[d] = s_new
                append!(s, getSplits(X,
                          lb,
                          upperBound,
                          minData,
                          ϵ,
                          K,
                          d,
                          depth = depth+1
                         ))
                K_ += 1
            end
            if (z1 > minData) && (K_ < K)
                ub = copy(upperBound)
                ub[d] = s_new
                append!(s, getSplits(X,
                          lowerBound,
                          ub,
                          minData,
                          ϵ,
                          K,
                          d,
                          depth = depth+1
                         ))
            end
        end

        push!(s, s_new)
    end
    return s
end

function _buildSplit(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::DSMGPConfig,
                    depth::Int,
                    observations::Vector{Int},
                    N::Int;
                    d = 1
                  )

    @assert all(isfinite.(X))

    l = max(lowerBound[d])
    u = min(upperBound[d])

    idx = findall((X[:,d] .> l) .& (X[:,d] .<= u))

    s = getSplits(X,
                  lowerBound,
                  upperBound,
                  config.minData,
                  config.bnoise,
                  config.K,
                  d)
    sort!(s)

    split = Vector{Tuple{Int, Float64}}()
    if !isempty(s)
        for si in s
            push!(split, (d, si))
        end
    end
    push!(split, (d, upperBound[d]))

    node = GPSplitNode(gensym("split"), Vector{Node}(), Vector{SPNNode}(),
                       lowerBound, upperBound, split)


    lb = copy(lowerBound)
    ub = copy(upperBound)

    if !isempty(s)
        for spliti in split
            (_, si) = spliti
            lb_ = copy(lb)
            ub_ = copy(ub)
            ub_[d] = si

            idx = findall((X[:,d] .> lb_[d]) .& (X[:,d] .<= ub_[d]))
            if (depth < config.depth) && (length(idx) > config.minData)
                if config.sumRoot
                    add!(node,
                         _buildSum(view(X,idx,:), view(y,idx), lb_, ub_, config, depth,
                                  observations[idx], N)
                        )
                else
                    add!(node,
                         _buildSplit(view(X,idx,:), view(y,idx), lb_, ub_, config, depth,
                                    observations[idx], N)
                        )
                end
            else
                add!(node, _buildGP(view(X,idx,:), view(y,idx), lb_, ub_, config,
                                    observations[idx], N))
            end
            lb[d] = si
        end
        return node
    else
        l = lowerBound[d]
        u = upperBound[d]

        idx = findall((X[:,d] .> l) .& (X[:,d] .<= u))

        return _buildGP(view(X,idx,:), view(y, idx), copy(lowerBound), copy(upperBound),
                        config, observations[idx], N)
    end
end

function _buildSum(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::DSMGPConfig,
                    depth::Int,
                    observations::Vector{Int},
                    N::Int;
                    d = 1
                  )
    @assert all(isfinite.(X))

    V = config.V
    w = fill(-log(V), V)
    node = GPSumNode{Float64,SPNNode}(gensym("sum"),
                              Vector{Node}(),
                              Vector{SPNNode}(),
                              Vector{Float64}())

    dims = collect(1:size(X,2))
    ϕ = map(d -> maximum(X[:,d])-minimum(X[:,d]), dims)
    ϕ ./= sum(ϕ)
    for v = 1:V
        d = rand(Categorical(ϕ))
        add!(node,
            _buildSplit(X, y, lowerBound, upperBound, config, depth+1,
                        observations, N, d = d), w[v]
            )
    end
    return node
end

function _buildGP(X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::DSMGPConfig,
                    observations::Vector{Int},
                    N::Int
                )
    myobs = falses(N)
    myobs[observations] .= true

    @assert size(X,1) == sum(myobs)

    if config.kernels isa Vector

        w = rand(Dirichlet(length(config.kernels), 1.0))
        node = GPSumNode{Float64,GPNode}(gensym("sum"),
                                         Vector{Node}(),
                                         Vector{GPNode}(),
                                         Vector{Float64}())

        for v in 1:length(config.kernels)
            kern = deepcopy(config.kernels[v])
            obsNoise = copy(config.observationNoise)

            # create a full GP
            mfun = config.meanFun == nothing ? ConstMean(mean(y)) : config.meanFun
            gp = GaussianProcess(X, y, kernel = kern, mean = mfun,
                                 logNoise = obsNoise, run_cholesky = false)

            add!(node, GPNode(gensym("GP"),
                              Vector{Node}(),
                              gp,
                              myobs,
                              observations,
                              lowerBound,
                              upperBound,
                              sum(myobs),
                              v
                             ), log(w[v]))
        end
        return node
    else
        kern = deepcopy(config.kernels)
        obsNoise = copy(config.observationNoise)

        # create a full GP
        mfun = config.meanFun == nothing ? ConstMean(mean(y)) : config.meanFun
        gp = GaussianProcess(X, y, kernel = kern, mean = mfun,
                                 logNoise = obsNoise, run_cholesky = false)

        return GPNode(gensym("GP"),
                      Vector{Node}(),
                      gp,
                      myobs,
                      observations,
                      lowerBound,
                      upperBound,
                      sum(myobs),
                      1
                     )
    end
end

"""
    buildDSMGP(x,y,K,V; ϵ=0.5, M=30, D=2, kernel=IsoSE(1.0,1.0), meanFun=nothing, logNoise=1.0, sum=true)

    Build a deep structured mixture of Gaussian processes (DSMGP).

Arguments:

* x: Observed input data (Matrix)
* y: Observed output data (Vector)
* K: Number of children under each sum node
* V: Number of splits at each split node
* ϵ: Split position noise parameter (higher means less data-driven splits)
* M: Minimum number of observations per expert
* D: Maximum depth of model
* kernel: Kernel function
* meanFun: Mean function (if nothing use independent ConstMean for each expert)
* logNoise: Log of the likelihood noise variance parameter
* sum: Use sum nodes
"""
function buildDSMGP(x,y,K::Int,V::Int;
               ϵ = 0.5,
               M = 30,
               D = 2,
               kernel = IsoSE(1.0, 1.0),
               meanFun = nothing,
               logNoise = 1.0,
               sum = true
              )
    m,D,gpmap = build(x,y,K,V,ϵ,M,D,kernel,meanFun,logNoise,sum)
    return DSMGP(m, D, gpmap)
end

"""
    buildPoE(x,y,V; ϵ=0.0, M=30, D=2, kernel=IsoSE(1.0,1.0), meanFun=nothing, logNoise=1.0, generalized=false)

Build (generalized) Product-of-Experts.

Arguments:

* x: Observed input data (Matrix)
* y: Observed output data (Vector)
* V: Number of splits at each split node
* ϵ: Split position noise parameter (higher means less data-driven splits)
* M: Minimum number of observations per expert
* D: Maximum depth of model
* kernel: Kernel function
* meanFun: Mean function (if nothing use independent ConstMean for each expert)
* logNoise: Log of the likelihood noise variance parameter
* generalized: Use generalized formulation (Deisenroth et al. 2015)

"""
function buildPoE(x,y,V::Int;
               ϵ = 0.0,
               M = 30,
               D = 2,
               kernel = IsoSE(1.0, 1.0),
               meanFun,
               logNoise = 1.0,
               generalized = false
              )
    m,D,gpmap = build(x,y,1,V,ϵ,M,D,kernel,meanFun,logNoise,false)
    return generalized ? gPoE(m, D, gpmap) : PoE(m, D, gpmap)
end

"""
    buildBCM(x,y,V; ϵ=0.0, M=30, D=2, kernel=IsoSE(1.0,1.0), meanFun=nothing, logNoise=1.0)

Build robust Bayesian Committee machine.

Arguments:

* x: Observed input data (Matrix)
* y: Observed output data (Vector)
* V: Number of splits at each split node
* ϵ: Split position noise parameter (higher means less data-driven splits)
* M: Minimum number of observations per expert
* D: Maximum depth of model
* kernel: Kernel function
* meanFun: Mean function (if nothing use independent ConstMean for each expert)
* logNoise: Log of the likelihood noise variance parameter
* robust: Use robust formulation (Deisenroth et al. 2015)

"""
function buildBCM(x,y,V::Int;
               ϵ = 0.0,
               M = 30,
               D = 2,
               kernel = IsoSE(1.0, 1.0),
               meanFun = nothing,
               logNoise = 1.0,
               robust = false
              )
    m,D,gpmap = build(x,y,1,V,ϵ,M,D,kernel,meanFun,logNoise,false)
    return rBCM(m, D, gpmap)
end

function build(x, y, K::Int, V::Int, ϵ, M, D, kernel, meanFun, logNoise, useSum)

    # DSMGP with a multiple independent GPs
    config = DSMGPConfig(
        meanFun,
        kernel, # kernel function / kernel functions
        logNoise, # log σ - Noise
        M, # max number of samples per sub-region
        V, # K = number of splits per split node (not used)
        K, # V = number of children under a sum node
        D, # maximum depth of the tree
        ϵ, # relative noise used to displace split positions
        useSum # use sum root
    )
    spn = buildTree(x, y, config);

    gpids = getLeafIds(spn)

    x = Dict(id[2] => id[1] for id in enumerate(gpids))
    fx = Dict(id[1] => id[2] for id in enumerate(gpids))

    gpmap = BiDict(x,fx)

    D = zeros(Float64, length(gpids), length(gpids));

    # update D
    getOverlap(spn, D, gpmap);

    # fit model
    fit!(spn,D,gpmap)

    return spn, D, gpmap
end
