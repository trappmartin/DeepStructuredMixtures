using DeepStructuredMixtures
using Random, LinearAlgebra
using Plots, StatsFuns

# create some random data
D = 2
N = 50

Random.seed!(123)
x = rand(N, D) 
y = cos.(6*(x[:,1] + x[:,2])) - 2*sin.(6*(x[:,1] + x[:,2]))

# plot the data
scatter(x[:,1], x[:,2], zcolor = y)

# learn a DSMGP
K = 2
V = 4
kernelf = IsoSE(log(0.1), log(0.1))
Random.seed!(123)

dsmgp = buildDSMGP(x, y, K, V; ϵ = 0.0, M = 100, kernel = kernelf, logNoise = log(0.1))
update!(dsmgp)

xrange = 0:0.05:1
yrange = xrange

f(x, y) = predict(dsmgp, [x y])[1]
Vf(x, y) = predict(dsmgp, [x y])[2]

μexact = mapreduce(yy -> mapreduce(xx -> f(xx, yy), hcat, xrange), vcat, yrange)
Σexact = mapreduce(yy -> mapreduce(xx -> Vf(xx, yy), hcat, xrange), vcat, yrange)

contour(xrange, yrange, μexact, fill=true)
scatter!(x[:,1], x[:,2])
savefig("fullGP_mean.png")

contour(xrange, yrange, Σexact, fill=true)
scatter!(x[:,1], x[:,2])
savefig("fullGP_cov.png")

M = 20


Random.seed!(123)
dsmgp = buildDSMGP(x, y, K, V; ϵ = 0.1, M = M, kernel = kernelf, logNoise = log(0.1))
#update!(dsmgp)

f(x, y) = predict(dsmgp, [x y])[1]
Vf(x, y) = predict(dsmgp, [x y])[2]

μ = mapreduce(yy -> mapreduce(xx -> f(xx, yy), hcat, xrange), vcat, yrange)
Σ = mapreduce(yy -> mapreduce(xx -> Vf(xx, yy), hcat, xrange), vcat, yrange)

contour(xrange, yrange, μ, fill=true)
scatter!(x[:,1], x[:,2], zcolor = y)

deviations = (round(sum(abs.(μexact - μ)), digits=3), round(sum(abs.(Σexact - Σ)), digits=3))

contour(xrange, yrange, μ, fill=true)
scatter!(x[:,1], x[:,2])
title!("μ DSMGP[M=$M]: deviation = $(deviations[1])")
savefig("DSMGP_$(M)_mean.png")

contour(xrange, yrange, Σ, fill=true)
scatter!(x[:,1], x[:,2])
title!("Σ DSMGP[M=$M]: deviation = $(deviations[2])")
savefig("DSMGP_$(M)_cov.png")


# partition the axis
function replaceGP!(n, d, x, y)
    replaceGP!.(children(n), d, Ref(x), Ref(y))
end

function replaceGP!(n::GPSplitNode, d, x, y)
    di, s = first(n.split)
    if d != di
        replaceGP!.(children(n), d, Ref(x), Ref(y))
    else
        for k in 1:length(n)-1
            deleteat!(n.children, 1)
            deleteat!(n.split, 1)
        end
        replaceGP!(children(n)[1], d, x, y)
    end
end

function replaceGP!(n::GPNode, d, x, y)
    N,D = size(x)
    dims = filter(di -> di != d, 1:D)

    f(xi, lb, ub) = (xi > lb) & (xi <= ub)
    b = map(i -> all(f(x[i,di], n.lb[di], n.ub[di]) for di in dims), 1:N)
    idx = findall(b)
    
    # create a full GP
    GaussianProcess(x[idx,:], y[idx]; 
        kernel = deepcopy(n.dist.kernel),
        mean = deepcopy(n.dist.mean),
        logNoise = n.dist.logNoise.value,
        run_cholesky = true
        )

    return length(idx) 
end

function _buildSplit(
                    X::AbstractMatrix,
                    y::AbstractVector,
                    lowerBound::Vector{Float64},
                    upperBound::Vector{Float64},
                    config::DSMGPConfig,
                    depth::Int,
                    observations::Vector{Int},
                    N::Int,
                    forgetDim::Int;
                    d = 1
                  )

    @assert all(isfinite.(X))
    @assert all( l < u for (l,u) in zip(lowerBound, upperBound))

    l = max(lowerBound[d])
    u = min(upperBound[d])

    idx = findall((X[:,d] .> l) .& (X[:,d] .<= u))

    s = DeepStructuredMixtures.getSplits(X,
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
                                  observations[idx], N, forgetDim)
                        )
                else
                    add!(node,
                         _buildSplit(view(X,idx,:), view(y,idx), lb_, ub_, config, depth,
                                    observations[idx], N, forgetDim)
                        )
                end
            else
                @info "build GP with N = $(length(idx)) obs."
                add!(node, DeepStructuredMixtures._buildGP(view(X,idx,:), view(y,idx), lb_, ub_, config,
                                    observations[idx], N))
            end
            lb[d] = si
        end
        return node
    else
        l = lowerBound[d]
        u = upperBound[d]

        idx = findall((X[:,d] .> l) .& (X[:,d] .<= u))

        @info "build GP with N = $(length(idx)) obs."
        return DeepStructuredMixtures._buildGP(view(X,idx,:), view(y, idx), copy(lowerBound), copy(upperBound),
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
                    N::Int,
                    forgetDim::Int;
                    d = 1
                  )
    @assert all(isfinite.(X))

    V = config.V
    w = fill(-log(V), V)
    node = GPSumNode{Float64,SPNNode}(gensym("sum"),
                              Vector{Node}(),
                              Vector{SPNNode}(),
                              Vector{Float64}())

    dims = filter(j -> j != forgetDim, collect(1:size(X,2)))
    ϕ = map(d -> maximum(X[:,d])-minimum(X[:,d]), dims)
    ϕ ./= sum(ϕ)
    for v = 1:V
        d = rand(DiscreteNonParametric(dims, ϕ))
        add!(node,
            _buildSplit(X, y, lowerBound, upperBound, config, depth+1,
                        observations, N, forgetDim, d = d), w[v]
            )
    end
    return node
end

function buildTreeV2(X::AbstractMatrix, y::AbstractVector, config::DSMGPConfig)

    N,D = size(X)
    @assert N == length(y)

    lowerBound = ones(D) * -Inf
    upperBound = ones(D) * Inf

    observations = collect(1:N)

    @assert all(isfinite.(X))

    w = fill(-log(D), D)
    root = GPSumNode{Float64,SPNNode}(gensym("sum"),
                              Vector{Node}(),
                              Vector{SPNNode}(),
                              Vector{Float64}())

    for forgetDim in 1:D
        add!(root,
            _buildSum(X, y, lowerBound, upperBound, config, 0, observations, N, forgetDim), w[forgetDim]
            )
    end
    return root
end

function buildDSMGPV2(x,y,K::Int,V::Int;
               ϵ = 0.5,
               M = 30,
               D = 2,
               kernel = IsoSE(1.0, 1.0),
               meanFun = nothing,
               logNoise = 1.0,
               sum = true
              )
    m,D,gpmap = buildV2(x,y,K,V,ϵ,M,D,kernel,meanFun,logNoise,sum)
    return DSMGP(m, D, gpmap)
end

function buildV2(x, y, K::Int, V::Int, ϵ, M, D, kernel, meanFun, logNoise, useSum)

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
    spn = buildTreeV2(x, y, config);

    gpids = DeepStructuredMixtures.getLeafIds(spn)

    x = Dict(id[2] => id[1] for id in enumerate(gpids))
    fx = Dict(id[1] => id[2] for id in enumerate(gpids))

    gpmap = DeepStructuredMixtures.BiDict(x,fx)

    D = zeros(Float64, length(gpids), length(gpids));

    # update D
    DeepStructuredMixtures.getOverlap(spn, D, gpmap);

    # fit model
    fit!(spn,D,gpmap)

    return spn, D, gpmap
end

Random.seed!(123)
dsmgp2 = buildDSMGPV2(x, y, K, V; ϵ = 0.1, M = M, kernel = kernelf, logNoise = log(0.1))

f(x, y) = predict(dsmgp2, [x y])[1] 
Vf(x, y) = predict(dsmgp2, [x y])[2] 

Σ = mapreduce(yy -> mapreduce(xx -> Vf(xx, yy), hcat, xrange), vcat, yrange)
μ = mapreduce(yy -> mapreduce(xx -> f(xx, yy), hcat, xrange), vcat, yrange)

deviations = (round(sum(abs.(μexact - μ)), digits=3), round(sum(abs.(Σexact - Σ)), digits=3))

contour(xrange, yrange, μ, fill=true)
scatter!(x[:,1], x[:,2])
title!("μ Sliced DSMGP[M=$M]: deviation = $(deviations[1])")
savefig("DSMGP_sliced_$(M)_mean.png")

contour(xrange, yrange, Σ, fill=true)
scatter!(x[:,1], x[:,2])
title!("Σ Sliced DSMGP[M=$M]: deviation = $(deviations[2])")
savefig("DSMGP_sliced_$(M)_cov.png")

