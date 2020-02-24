export update!, reset_weights!, infer!
export getLogNoise, kernelid
export blockmatrix, blockindecies
export nummixtures

nummixtures(node::GPNode) = 1
nummixtures(node::GPSplitNode) = mapreduce(nummixtures, *, children(node))
nummixtures(node::GPSumNode) = mapreduce(nummixtures, +, children(node))


function blockmatrix(node::GPNode)
    M = zeros(length(node.observations), length(node.observations))
    idx = findall(node.observations)
    M[idx, idx] .+= 1
    return M
end

function blockmatrix(node::Node)
    M = blockmatrix(node[1])
    for k = 2:length(node)
        M .+= blockmatrix(node[k])
    end
    return M
end

function blockmatrix(node::GPSumNode)
    M = weights(node)[1]*blockmatrix(node[1])
    for k = 2:length(node)
        M .+= weights(node)[k]*blockmatrix(node[k])
    end
    return M
end


function blockindecies(node::GPNode, Ix::Vector{Vector{Int}})
    for n in node.obs
        append!(Ix[n], node.obs)
    end
end

function blockindecies(node::Node, Ix::Vector{Vector{Int}})
    return map(child -> blockindecies(child, Ix), children(node))
end

bestblockmatrix(node::GPNode) = blockmatrix(node)
function bestblockmatrix(node::Node)
    mapreduce(bestblockmatrix, +, children(node))
end

function bestblockmatrix(node::GPSumNode)
    i = argmax(node.logweights)
    return bestblockmatrix(node[i])
end

@inline kernelid(node::GPNode, x::AbstractMatrix) = repeat([node.kernelid], size(x,1))
function kernelid(node::GPSplitNode, x::AbstractMatrix)
    idx = getchild(node, x)
    kernel = zeros(Int, size(x,1))
    for (k, c) in enumerate(children(node))
        j = findall(idx .== k)
        if !isempty(j)
            kernel_ = kernelid(c, x[j,:])
            kernel[j] = kernel_
        end
    end
    return kernel
end

function kernelid(node::GPSumNode{T,V}, x::AbstractMatrix) where {T,V<:SPNNode}
    w = weights(node)
    k_ = mapreduce(c -> kernelid(c,x), hcat, children(node))
    uk = unique(k_)
    c = mapreduce(kk -> sum((k_ .== kk) .* w', dims=2), hcat, uk)
    kernel = map(i -> uk[i[2]], argmax(c, dims=2))
    return kernel
end

function kernelid(node::GPSumNode{T,V}, x::AbstractMatrix) where {T,V<:GPNode}
    i = argmax(node.logweights)
    kernel = kernelid(node[i], x)
    return kernelid(node[i], x)
end

getLogNoise(node::GPNode, x::AbstractMatrix) = repeat([node.dist.logNoise.value], size(x,1))
function getLogNoise(node::GPSplitNode, x::AbstractMatrix)
    idx = getchild(node, x)
    noise = zeros(size(x,1))
    for (k, c) in enumerate(children(node))
        j = findall(idx .== k)
        noise_ = getLogNoise(c, x[j,:])
        noise[j] = noise_
    end
    return noise
end

function getLogNoise(node::GPSumNode, x::AbstractMatrix)
    return lse(mapreduce(k -> node.logweights[k].+getLogNoise(node[k],x), hcat, 1:length(node)))
end


function getchild(node::GPSplitNode, x::AbstractMatrix)
    idx = zeros(Int, size(x,1))
    @inbounds for n in 1:size(x,1)
        k = 1
        while idx[n] == 0
            split = node.split[k]
            d, s = split

            accept = if k == 1
                x[n,d] <= s
            else
                (x[n,d] <= s) & (x[n,d] > node.split[k-1][2])
            end

            if accept
                idx[n] = k
            end
            k += 1
        end
    end
    return idx
end

leftGP(node::GPSumNode{T,C}) where {T<:Real,C<:SPNNode} = leftGP(first(children(node)))
leftGP(node::GPSumNode{T,GPNode}) where {T<:Real} = mapreduce(leftGP, vcat, children(node))
leftGP(node::GPSplitNode) = leftGP(first(children(node)))
leftGP(node::GPNode) = node.dist

rightGP(node::GPSumNode{T,C}) where {T<:Real,C<:SPNNode} = rightGP(last(children(node)))
rightGP(node::GPSumNode{T,GPNode}) where {T<:Real} = mapreduce(rightGP, vcat, children(node))
rightGP(node::GPSplitNode) = rightGP(last(children(node)))
rightGP(node::GPNode) = node.dist

function _predict(node::GPNode, x::AbstractMatrix, μmin)
    μ, Σ = prediction(node.dist, x)
    σ² = diag(Σ)
    σ²[σ² .<= 0] .= ϵ
    @assert all(μ .>= μmin)
    lm = log.(μ-μmin)
    lm2 = log.(μ.^2)
    ls = log.(σ²)
    return lm, lm2, ls, ones(Int, size(x,1))
end

function _predictPoE(node::GPNode, x::AbstractMatrix)
    μ, Σ = prediction(node.dist, x)
    σ² = diag(Σ)
    return μ, inv.(σ²)
end

function _minpredict(node::GPNode, x::AbstractMatrix)
    μ, _ = prediction(node.dist, x)
    return μ
end

function _minpredict(node::GPSplitNode, x::AbstractMatrix)
    idx = getchild(node, x)
    μ = zeros(size(x,1))
    for (k, c) in enumerate(children(node))
        j = findall(idx .== k)
        μ_ = _minpredict(c, x[j,:])
        μ[j] = μ_
    end
    return μ
end

function _minpredict(node::GPSumNode, x::AbstractMatrix)
    μ = ones(size(x,1)) * Inf
    for (k, c) in enumerate(children(node))
        μ = vec(minimum([μ _minpredict(c, x)], dims=2))
    end
    return μ
end

function predict(node::GPNode, x::AbstractMatrix)
    μmin = _minpredict(node, x)
    lμ, _, lwσ², _ = _predict(node, x, μmin .- 1)
    return exp.(lμ) + μmin .- 1, exp.(lwσ²)
end

function _predict(node::GPSplitNode, x::AbstractMatrix, μmin)
    idx = getchild(node, x)
    lμ = zeros(size(x,1))
    lwμ² = zeros(size(x,1))
    lwσ² = zeros(size(x,1))
    n = zeros(Int, size(x,1))
    for (k, c) in enumerate(children(node))
        j = findall(idx .== k)
        lμ_, lwμ²_, lwσ²_, n_ = _predict(c, x[j,:], μmin[j])
        lμ[j] = lμ_
        lwμ²[j] = lwμ²_
        lwσ²[j] = lwσ²_
        n[j] = n_
    end
    return lμ, lwμ², lwσ², n
end

function _predictPoE(node::GPSplitNode, x::AbstractMatrix)
    μ = zeros(size(x,1))
    t = zeros(size(x,1))

    for (k,c) in enumerate(children(node))
        μ_, t_ = _predictPoE(c, x)
        t[:] += t_
        μ[:] += t_ .* μ_
    end
    return μ ./ t, t
end

# same as for PoE
function _predictgPoE(node::GPSplitNode, x::AbstractMatrix, M::Int)
    μ = zeros(size(x,1))
    t = zeros(size(x,1))
    M = length(node)
    β = 1/M
    for (k,c) in enumerate(children(node))
        μ_, t_ = _predictPoE(c, x)
        t[:] += β*t_
        μ[:] += β*t_ .* μ_
    end
    return μ ./ t, t
end

function _predictrBCM(node::GPSplitNode, x::AbstractMatrix)
    μ = zeros(size(x,1))

    gp = leftGP(node)
    s = diag(kernelmatrix(gp.kernel, x, x)) .+ getnoise(gp)

    C = deepcopy(1 ./ s)

    for (k,c) in enumerate(children(node))
        μ_, t_ = _predictPoE(c, x)
        s_ = 1 ./ t_
        β_ = 0.5 * (log.(s) - log.(s_))
        C += (β_ .* t_) - (β_ ./ s)
        μ += μ_ .* (β_ .* t_)
    end

    return μ ./ C, C
end

function predict(node::GPSplitNode, x::AbstractMatrix)
    idx = getchild(node, x)
    μ = zeros(size(x,1))
    σ² = zeros(size(x,1))
    for (k, c) in enumerate(children(node))
        j = findall(idx .== k)
        m, s = predict(c, x[j,:])
        μ[j] = m
        σ²[j] = s
    end
    return μ, σ²
end

function predictPoE(node::GPSplitNode, x::AbstractMatrix)
    μ, t = _predictPoE(node, x)
    σ² = inv.(t)
    return μ, σ²
end

# We use β = 1/M as described in Deisenroth et al. (ICML 2015)
function predictgPoE(node::GPSplitNode, x::AbstractMatrix)
    μ, t = _predictgPoE(node, x, 1)
    σ² = inv.(t)
    return μ, σ²
end

function predictrBCM(node::GPSplitNode, x::AbstractMatrix)
    μ, t = _predictrBCM(node, x)
    σ² = inv.(t)
    return μ, σ²
end

function _predict(node::GPSumNode, x::AbstractMatrix, μmin)

    lμ = zeros(size(x,1), length(node))
    lwμ² = zeros(size(x,1), length(node))
    lwσ² = zeros(size(x,1), length(node))
    n = zeros(Int, size(x,1))

    for (k, c) in enumerate(children(node))
        lμ_, lwμ²_, lwσ²_, n_ = _predict(c, x, μmin)

        lμ[:,k] = lμ_ .+ logweights(node)[k]
        lwμ²[:,k] = lwμ²_ .+ logweights(node)[k]
        lwσ²[:,k] = lwσ²_ .+ logweights(node)[k]
        n += n_
    end

    return vec(lse(lμ)), vec(lse(lwμ²)), vec(lse(lwσ²)), n
end

function predict(node::GPSumNode, x::AbstractMatrix)

    μmin = _minpredict(node, x)

    lμ, lwμ², lwσ², n = _predict(node, x, μmin .- 1)
    μ = exp.(lμ) + μmin .- 1
    v = exp.(lwσ²) + (exp.(lwμ²) - μ.^2)
    return μ, v
end

@inline predict(model::DSMGP, x::AbstractMatrix) = predict(model.root, x)
@inline predict(model::PoE, x::AbstractMatrix) = predictPoE(model.root, x)
@inline predict(model::gPoE, x::AbstractMatrix) = predictgPoE(model.root, x)
@inline predict(model::rBCM, x::AbstractMatrix) = predictrBCM(model.root, x)

function lse(x::AbstractMatrix{<:Real}; dims = 2)
    m = maximum(x, dims = dims)
    v = exp.(x .- m)
    return log.(sum(v, dims = dims)) + m
end

@inline getx(node::GPNode) = node.dist.x
@inline getx(node::GPSplitNode) = mapreduce(c -> getx(c), vcat, children(node))
@inline getx(node::GPSumNode) = getx(node[1])

@inline gety(node::GPNode) = node.dist.y + get(node.dist.mean, node.dist.N)
@inline gety(node::GPSplitNode) = mapreduce(c -> gety(c), vcat, children(node))
@inline gety(node::GPSumNode) = gety(node[1])

@inline update!(node::GPNode) = mll(node.dist)
@inline update!(node::GPSplitNode) = mapreduce(update!, +, children(node))

function update!(node::GPSumNode)
    K = length(node)
    map!(c -> -log(K)+update!(c), node.logweights, children(node))
    z = StatsFuns.logsumexp(node.logweights)
    map!(lw -> lw - z, node.logweights, node.logweights)
    return z
end

@inline update!(spn::DSMGP) = update!(spn.root)

@inline infer!(node::GPNode) = mll(node.dist)
@inline infer!(node::GPSplitNode) = mapreduce(infer!, +, children(node))

function infer!(node::GPSumNode{T,V}) where {T<:AbstractFloat,V<:GPNode}
    K = length(node)
    map!(c -> -log(K)+infer!(c), node.logweights, children(node))
    z = StatsFuns.logsumexp(node.logweights)
    map!(lw -> lw - z, node.logweights, node.logweights)
    return z
end

function infer!(node::GPSumNode{T,V}) where {T<:AbstractFloat,V<:SPNNode}
    K = length(node)
    map!(c -> -log(K)+infer!(c), node.logweights, children(node))
    z = StatsFuns.logsumexp(node.logweights)
    fill!(node.logweights, -log(K))
    return z
end

@inline infer!(spn::DSMGP) = infer!(spn.root)

function reset_weights!(spn::DSMGP)
    snodes = filter(n -> n isa SumNode, SumProductNetworks.getOrderedNodes(spn.root))
    for n in snodes
        K = length(n)
        fill!(n.logweights, -log(K))
    end
end

function stats(node::GPNode; dict::Dict{Symbol,Any} = Dict{Symbol,Any}())
    dict[:gps] = get(dict, :gps, 0) + 1
    if !haskey(dict, :ndata)
        dict[:ndata] = Vector{Int}()
    end
    push!(dict[:ndata], node.dist.N)
end

function stats(node::GPSumNode; dict::Dict{Symbol,Any} = Dict{Symbol,Any}())
    for c in children(node)
        stats(c, dict = dict)
    end
    dict[:sumnodes] = get(dict, :slitnodes, 0) + 1
    return dict
end

function stats(node::GPSplitNode; dict::Dict{Symbol,Any} = Dict{Symbol,Any}())
    for c in children(node)
        stats(c, dict = dict)
    end
    dict[:slitnodes] = get(dict, :slitnodes, 0) + 1
    if !haskey(dict, :bounds)
        dict[:bounds] = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
    end
    push!(dict[:bounds], (node.lowerBound, node.upperBound))
    if !haskey(dict, :ids)
        dict[:ids] = Vector{Symbol}()
    end
    push!(dict[:ids], node.id)
    return dict
end
