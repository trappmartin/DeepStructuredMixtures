using Base.Threads
using DeepGaussianProcessExperts.AdvancedCholesky

export getOverlap, updateK!, fit!, fit_naive!
export getLeafIds

@inline getLeafIds(node::GPNode) = Dict{Symbol,GPNode}(node.id => node)
@inline getLeafIds(node::GPSplitNode) = mapreduce(getLeafIds, merge, children(node))
@inline getLeafIds(node::GPSumNode) = mapreduce(getLeafIds, merge, children(node))

@inline getOverlap(node::GPNode, D::Matrix{T}, ids) where {T<:Real} = Dict(findfirst(node.id .== ids) => node)
@inline function getOverlap(node::GPSplitNode, D::Matrix{T}, ids) where {T<:Real}
    return mapreduce(c -> getOverlap(c, D, ids), merge, children(node))
end
function getOverlap(node::GPSumNode, D::Matrix{T}, ids) where {T<:Real}
    r = map(c -> getOverlap(c, D, ids), children(node))
    @inbounds begin
        for i = 1:length(r)
            for j = (i+1):length(r)
                for n in keys(r[i])
                    nnode = r[i][n]
                    for m in keys(r[j])
                        mnode = r[j][m]
                        Δ = xor.(nnode.observations, mnode.observations)
                        Δn = sum(Δ .& nnode.observations)
                        Δm = sum(Δ .& mnode.observations)
                        D[n, m] = one(T) - T(Δn / sum(nnode.observations))
                        D[m, n] = one(T) - T(Δm / sum(mnode.observations))
                    end
                end
            end
        end
    end

    return reduce(merge, r)
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

function fit!(spn::Union{GPSumNode, GPSplitNode}, D::Matrix; τ = 0.2)

    # total time taken for Cholesky decompositions
    ttotal = 0

    gpmapping = getLeafIds(spn)
    gpids = collect(keys(gpmapping))
    K = length(gpids)

    # re-scale by number of observations per region
    l = map(k -> Float64(sum(gpmapping[k].observations)), gpids)
    l ./= maximum(l)

    solvedIds = Set{Symbol}()

    P = sum(D .* l, dims=2)

    ids = sort(collect(1:length(gpids)), by = (i) -> P[i], rev=true)

    obs = Vector{Vector{Int}}(undef, length(ids))
    processed = Vector{Threads.AbstractLock}(undef, length(ids))
    queued = Vector{Int}()
    for i in ids
        accept = all(j -> D[i,j] == 0, queued)
        if accept
            push!(queued, i)
        end
        obs[i] = findall(gpmapping[gpids[i]].observations)
        processed[i] = SpinLock()
    end

    ttotal = @elapsed for i in queued

        mainNode = gpmapping[gpids[i]]
        mainGP = mainNode.dist

        # solve the main GP
        update_mll!(mainGP)

        _ids = sort(ids, by = (j) -> D[i,j], rev=true)

        for j in filter(j -> j ∉ queued, _ids)
            if trylock(processed[j])

                jNode = gpmapping[gpids[j]]
                jGP = jNode.dist

                if D[i,j] == D[j,i] == one(eltype(D))
                    # copy Cholesky
                    jGP.cK.mat[:] = mainGP.cK.mat
                    jGP.cK.chol.factors[:] = mainGP.cK.chol.factors
                    jGP.alpha[:] = mainGP.alpha
                    jGP.mll = mainGP.mll
                elseif (D[i,j] == zero(eltype(D))) || !(jNode.firstobs >= mainNode.firstobs)
                    # solve Cholesky
                    update_mll!(jGP)
                elseif D[j,i] == one(eltype(D))
                    # j is a sub-region or overlaps
                    if jNode.firstobs == mainNode.firstobs
                        # easy case
                        # Unfrequent situation
                        #observations_ = @inbounds filter(n -> obsj[obs[i][n]], 1:length(obs[i]))
                        idcs = 1:jNode.nobs
                        begin
                            jGP.cK.mat[:] = mainGP.cK.mat[idcs,idcs]
                            jGP.cK.chol.factors[:] = mainGP.cK.chol.factors[idcs,idcs]
                            update_mll!(jGP, kern=false, domean=false, noise=false)
                        end
                    else
                        # solve with low-rank update
                        # Most frequent situation
                        toupdate = 1:(jNode.firstobs-mainNode.firstobs)

                        # only do low-rank updates if sufficiently stable
                        if (length(toupdate) / jNode.nobs) < τ
                            CC = copy(mainGP.cK.chol.factors)
                            d = size(CC,1)
                            for n in toupdate
                                AdvancedCholesky.lowrankupdate!(CC,
                                                                view(CC,n,(n+1):d),
                                (n+1),
                                mainGP.cK.chol.uplo)
                            end
                            N = length(toupdate)+1
                            M = jNode.nobs-1
                            jGP.cK.mat[:] = mainGP.cK.mat[N:(N+M),N:(N+M)]
                            jGP.cK.chol.factors[:] = CC[N:(N+M),N:(N+M)]
                            update_mll!(jGP, kern=false, domean=false, noise=false)
                        else
                            # solve Cholesky
                            update_mll!(jGP)
                        end
                    end
                else
                    Σbuffer = GaussianProcesses.mat(jGP.cK)
                    GaussianProcesses.cov!(Σbuffer, jGP.kernel, jGP.x, jGP.x, jGP.data)
                    noise = eltype(Σbuffer)(exp(2*jGP.logNoise)+eps())
                    @inbounds Σbuffer[diagind(Σbuffer)] .+= noise

                    F = jGP.cK.chol.factors

                    # j isa larger than main region
                    if jNode.firstobs == mainNode.firstobs
                        # easy case
                        # Unfrequent situation
                        @inbounds F[1:mainNode.nobs,1:mainNode.nobs] = mainGP.cK.chol.factors
                        AdvancedCholesky.chol_continue(F, mainNode.nobs+1)
                        begin
                            update_mll!(jGP, kern=false, domean=false, noise=false)
                        end
                    else
                        # solve with low-rank update & continue computation
                        # Most frequent situation
                        toupdate = 1:(jNode.firstobs-mainNode.firstobs)

                        # only do low-rank updates if sufficiently stable
                        if (length(toupdate) / jNode.nobs) < τ
                            CC = copy(mainGP.cK.chol.factors)
                            d = size(CC,1)
                            @inbounds for n in toupdate
                                AdvancedCholesky.lowrankupdate!(CC,
                                                                view(CC,n,(n+1):d),
                                                                (n+1),
                                                                mainGP.cK.chol.uplo)
                            end

                            Ni = length(toupdate)+1
                            N = mainNode.nobs - length(toupdate)
                            @inbounds F[1:N,1:N] = CC[Ni:end,Ni:end]
                            info = AdvancedCholesky.chol_continue!(F, N+1)
                            if info == 0
                                update_mll!(jGP, kern=false, domean=false, noise=false)
                            else
                                update_mll!(jGP)
                            end
                        else
                            # solve Cholesky
                            update_mll!(jGP)
                        end
                    end
                end
            end
        end
    end

    ttotal
end

function fit_naive!(spn::Union{GPSplitNode,GPSumNode})

    gpmapping = getLeafIds(spn)
    gpids = collect(keys(gpmapping))
    K = length(gpids)
    ttotal = @elapsed for gpid in gpids
        update_mll!(gpmapping[gpid].dist)
    end
    #   @info "[fit_naive!] finished with $ttotal sec taken for Cholesky decompositions"
    ttotal
end
