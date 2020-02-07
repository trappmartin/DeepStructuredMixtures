using Base.Threads
using DeepStructuredMixtures.AdvancedCholesky

export getOverlap, updateK!, fit!, fit_naive!
export updategradients!
export getLeaves
export distancematrix, kernelmatrix

@inline getLeaves(node::GPNode) = [node]
@inline getLeaves(node::Node) = mapreduce(getLeaves, vcat, children(node))

@inline function getOverlap(node::GPNode, D::Matrix{T}, idmap::BiDict) where {T<:Real}
    return [node]
end
@inline function getOverlap(node::GPSplitNode, D::Matrix{T}, idmap::BiDict) where {T<:Real}
    return mapreduce(c -> getOverlap(c, D, idmap), vcat, children(node))
end
function getOverlap(node::GPSumNode, D::Matrix{T}, idmap::BiDict) where {T<:Real}
    r = map(c -> getOverlap(c, D, idmap), children(node))
    @inbounds begin
        for i = 1:length(r)
            for j = (i+1):length(r)
                for nnode in r[i]
                    n = idmap.x[nnode.id]
                    for mnode in r[j]
                        m = idmap.x[mnode.id]
                        Δ = xor.(nnode.observations, mnode.observations)
                        Δn = sum(Δ .& nnode.observations) * (nnode.kernelid == mnode.kernelid)
                        Δm = sum(Δ .& mnode.observations) * (nnode.kernelid == mnode.kernelid)
                        D[n,m] = one(T) - T(Δn / sum(nnode.observations))
                        D[m,n] = one(T) - T(Δm / sum(mnode.observations))
                    end
                end
            end
        end
    end

    return reduce(vcat, r)
end

function getObservationCount!(node::GPNode, P::Matrix{Int})
    for n in node.observations
        for m in node.observations
            if n != m
                P[n,m] += 1
            end
        end
    end
end
@inline function getObservationCount!(node::GPSplitNode, P)
    map(c -> getObservationCount!(c, P), children(node))
end
@inline function getObservationCount!(node::GPSumNode, P)
    map(c -> getObservationCount!(c, P), children(node))
end

"""
    fit!(model::DSMGP; τ = 0.05)

Update the Cholesky decompositions of the DSMGP.

# Arguments:

* `τ`: Minimal relative overlap required to use shared computation. Higher values will lead to higher inaccuracies in the solutions.

"""
function fit!(spn::DSMGP; τ = 0.05)
    return fit!(spn.root, spn.D, spn.gpmap, τ = τ)
end

function fit!(spn::Union{GPSumNode, GPSplitNode}, D::Matrix, gpmap::BiDict; τ = 0.05)

    leaves = getLeaves(spn)
    n = length(leaves)
    processed = falses(n)
    counts = zeros(Int,n)
    S = Vector{GPNode}(undef,n)
    for j in 1:n
        i = argmax(D[:,j] .* D[j,:])
        counts[i] += 1
        nid = gpmap.fx[i]
        S[j] = leaves[findfirst(map(n -> n.id == nid, leaves))]
        #S[j] = findfirst(n.id == nid for n in leaves)
    end

    sort!(leaves, by = (node) -> counts[gpmap.x[node.id]])

    ttotal = @elapsed for jNode in leaves #for (ii, i) in enumerate(queued)
        j = gpmap.x[jNode.id]
        if !processed[j]

            mainNode = S[j]
            i = gpmap.x[mainNode.id]
            mainGP = mainNode.dist

            # solve the main GP
            if !processed[i]
                update_cholesky!(mainGP)
                processed[i] = true
            end

            jGP = jNode.dist
            processed[j] = true

            update_cholesky!(jGP)

            if mainNode.kernelid != jNode.kernelid
                # solve Cholesky
                update_cholesky!(jGP)
            elseif first(jNode.obs) < first(mainNode.obs)
                # solve Cholesky
                update_cholesky!(jGP)
            else
                ione = D[i,j] == one(eltype(D))
                jone = D[j,i] == one(eltype(D))
                fitcontained!(jNode, jGP, mainNode, mainGP, Val(ione), Val(jone), τ)
            end
        end
    end

    ttotal
end

function fitcontained!(jNode::GPNode,
                       jGP::GaussianProcess,
                       mainNode::GPNode,
                       mainGP::GaussianProcess,
                       ione, jone, τ::Float64)
    update_cholesky!(jGP)
end

function fitcontained!(jNode::GPNode,
                       jGP::GaussianProcess,
                       mainNode::GPNode,
                       mainGP::GaussianProcess,
                       ione::Val{true},
                       jone::Val{true},
                       τ::Float64
                      )
    # copy Cholesky
    jGP.cK.factors[:] = mainGP.cK.factors
    jGP.α[:] = mainGP.α
end

function fitcontained!(jNode::GPNode,
                       jGP::GaussianProcess,
                       mainNode::GPNode,
                       mainGP::GaussianProcess,
                       ione::Val{false},
                       jone::Val{true},
                       τ::Float64
                      )


    # j is a sub-region or overlaps

    # solve with low-rank update
    minJ = minimum(jNode.obs)
    maxJ = maximum(jNode.obs)

    minM = minimum(mainNode.obs)
    maxM = maximum(mainNode.obs)

    @assert minJ >= minM
    @assert maxJ <= maxM

    s = minJ == minM ? 1 : findfirst(mainNode.obs .== minJ)
    e = maxJ == maxM ? mainNode.nobs : findfirst(mainNode.obs .== maxJ)

    idx = collect(s:e)
    toupdate = setdiff(mainNode.obs[1:e], jNode.obs)

    # only do low-rank updates if sufficiently stable
    if (length(toupdate) / jNode.nobs) < τ

        CC = copy(mainGP.cK.factors)
        d = size(CC,1)

        for n in toupdate
            @inbounds begin
                i = findfirst(mainNode.obs .== n)
                AdvancedCholesky.lowrankupdate!(CC,
                                            view(CC,i,(i+1):d),
                                            (i+1),
                                            mainGP.cK.uplo)
            end
        end

        reverse!(toupdate)
        for n in toupdate
            i = findfirst(mainNode.obs[idx] .== n)
            !isnothing(i) && deleteat!(idx,i)
        end

        jGP.cK.factors[:] = CC[idx,idx]

        if all(diag(jGP.cK.factors) .>= 0)
            jGP.α[:] = jGP.cK.L' \ (jGP.cK.L \ jGP.y)
        else
            update_cholesky!(jGP)
        end
    else
        # solve Cholesky
        update_cholesky!(jGP)
    end
end

function fitcontained!(jNode::GPNode,
                       jGP::GaussianProcess,
                       mainNode::GPNode,
                       mainGP::GaussianProcess,
                       ione::Val{true},
                       jone::Val{false},
                       τ::Float64
                      )


    Knn = kernelmatrix(jGP.kernel, jGP.P)

    # reset factors to kernel matrix
    F = jGP.cK.factors
    Tchol = eltype(F)
    @inbounds F[:] = Tchol.(Knn)

    # compute noise
    noise = Tchol(getnoise(jGP) + ϵ)

    # add noise
    σ = @view F[diagind(F)]
    map!(i -> i+noise, σ, σ)

    # j isa larger than main region
    minJ = minimum(jNode.obs)
    maxJ = maximum(jNode.obs)

    minM = minimum(mainNode.obs)
    maxM = maximum(mainNode.obs)

    @assert minJ >= minM
    @assert maxJ >= maxM

    s = minJ == minM ? 1 : findfirst(mainNode.obs .== minJ)
    e = mainNode.nobs

    idx = collect(s:e)

    @inbounds s1 = jNode.obs[1:findfirst(jNode.obs .== maxM)]
    @inbounds s2 = mainNode.obs[idx]
    @inbounds toupdate = setdiff(mainNode.obs[1:e], jNode.obs[1:findfirst(jNode.obs .== maxM)])

    if (length(s1) != length(s2)) && (minJ == minM)
        update_cholesky!(jGP)
    else

        # only do low-rank updates if sufficiently stable
        if (length(toupdate) / jNode.nobs) < τ
            CC = copy(mainGP.cK.factors)
            d = size(CC,1)

            for n in toupdate
                @inbounds begin
                    i = findfirst(mainNode.obs .== n)
                    AdvancedCholesky.lowrankupdate!(CC,
                                                view(CC,i,(i+1):d),
                                                (i+1),
                                                mainGP.cK.uplo)
                end
            end

            reverse!(toupdate)
            for n in toupdate
                i = findfirst(mainNode.obs[idx] .== n)
                !isnothing(i) && deleteat!(idx,i)
            end

            @inbounds F[1:length(s1), 1:length(s1)] = CC[idx,idx]

            _,info = AdvancedCholesky.chol_continue!(F, length(s1)+1)

            check = all(diag(F) .>= 0.0)

            if (info == 0) && check
                @inbounds jGP.α[:] = jGP.cK.L' \ (jGP.cK.L \ jGP.y)
            else
                update_cholesky!(jGP)
            end
        else
            # solve Cholesky
            update_cholesky!(jGP)
        end
    end
end

function fit_naive!(spn::Union{GPSplitNode,GPSumNode})

    gpmapping = getLeaves(spn)
    gpids = collect(keys(gpmapping))
    K = length(gpids)
    ttotal = @elapsed for gpid in gpids
        update_cholesky!(gpmapping[gpid].dist)
    end
    #   @info "[fit_naive!] finished with $ttotal sec taken for Cholesky decompositions"
    ttotal
end

function updategradients!(spn::Union{GPSumNode, GPSplitNode})
    leaves = getLeaves(spn)
    Threads.@threads for leaf in leaves
        updategradients!(leaf.dist)
    end
end

function distancematrix(spn, kernel::IsoKernel, x::AbstractMatrix)
    N = length(gety(spn))
    Ix = map(n -> Vector{Int}(), 1:N)
    blockindecies(spn, Ix)
    V = map( i -> vec(getdistancematrix(kernel, reshape(x[i,:], 1, :), x[Ix[i],:])), 1:N)
    return SDiagonal(Ix, V)
end

function distancematrix(spn, kernel::ArdKernel, x::AbstractMatrix)
    N = length(gety(spn))
    Ix = map(n -> Vector{Int}(), 1:N)
    blockindecies(spn, Ix)
    V = map( i -> dropdims(getdistancematrix(kernel, reshape(x[i,:], 1, :), x[Ix[i],:]), dims=1), 1:N)
    return SDiagonal(Ix,V)
end

function updategradients!(spn::Union{GPSumNode, GPSplitNode},
                          K::SDiagonal{Tp,2,<:AbstractVector},
                          P::SDiagonal{Tp,2,<:AbstractVector},
                          D::AbstractMatrix{T},
                          gpmap) where {T,Tp,MTp}
    leaves = getLeaves(spn)
    n = length(leaves)
    isprocessed = falses(n)

    S = Dict{Int, Int}()
    for j in 1:n
        n = gpmap.x[leaves[j].id]
        m = argmax(D[:,n] .* D[n,:])
        if (D[n,m] * D[m,n]) == one(T)
            mid = gpmap.fx[m]
            S[j] = findfirst(map(n -> n.id == mid, leaves))
        end
    end

    for (j, leaf) in enumerate(leaves)
        if haskey(S, j)
            i = S[j]
            mainNode = leaves[i]
            if isprocessed[i]
                copygradients(leaf.dist, mainNode.dist)
            end
        else
            ix = leaf.obs
            updategradients!(leaf.dist, @view(K[ix,ix]), @view(P[ix,ix]))
        end
        isprocessed[j] = true
    end
end

function updategradients!(spn::Union{GPSumNode, GPSplitNode},
                          K::SDiagonal{Tp,2,<:AbstractVector},
                          P::SDiagonal{Tp,3,<:AbstractArray},
                          D::AbstractMatrix{T},
                          gpmap) where {T,Tp,MTp}
    leaves = getLeaves(spn)
    n = length(leaves)
    isprocessed = falses(n)

    S = Dict{Int, Int}()
    for j in 1:n
        n = gpmap.x[leaves[j].id]
        m = argmax(D[:,n] .* D[n,:])
        if (D[n,m] * D[m,n]) == one(T)
            mid = gpmap.fx[m]
            S[j] = findfirst(map(n -> n.id == mid, leaves))
        end
    end

    for (j, leaf) in enumerate(leaves)
        if haskey(S, j)
            i = S[j]
            mainNode = leaves[i]
            if isprocessed[i]
                copygradients(leaf.dist, mainNode.dist)
            end
        else
            ix = leaf.obs
            updategradients!(leaf.dist, @view(K[ix,ix]), @view(P[ix,ix,:]))
        end
        isprocessed[j] = true
    end
end
