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

function updateK!(node::GPNode, obs::Vector{Int})

    # change order of observations if necessary
    sdiff = setdiff(node.observations, obs)
    if !isempty(sdiff)
        # change ordering
        @info "Reordering ", length(sdiff), " observations from ", length(node.observations)
        @info length(obs)
    end

    gp = node.dist
    nobs = length(node.observations)
    Σbuffer = GaussianProcesses.mat(gp.cK)
    GaussianProcesses.cov!(Σbuffer, gp.kernel, gp.x, gp.x, gp.data)

    noise = exp(2*gp.logNoise)+eps()
    @inbounds for i in 1:nobs
        Σbuffer[i,i] += noise
    end

    C, info = LAPACK.potrf!(U', Σbuffer)
    @assert info == 0

    chol = Cholesky(C, 'U', info)

    gp.cK = PDMat(Σbuffer, chol)
    return chol
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
    mins = Vector{Int}(undef, length(ids))
    for i in ids
        accept = all(j -> D[i,j] == 0, queued)
        if accept
            push!(queued, i)
        end
        obs[i] = findall(gpmapping[gpids[i]].observations)
        processed[i] = SpinLock()
        mins[i] = findfirst(gpmapping[gpids[i]].observations)
    end

    for i in queued

        mainGP = gpmapping[gpids[i]].dist
        obsi = gpmapping[gpids[i]].observations

        # solve the main GP
        update_mll!(mainGP)

        _ids = sort(ids, by = (j) -> D[i,j], rev=true)

        for j in _ids
            if j ∈ queued
                continue
            end

            if !trylock(processed[j])
                continue
            end

            jGP = gpmapping[gpids[j]].dist
            obsj = gpmapping[gpids[j]].observations

            if D[i,j] == D[j,i] == one(eltype(D))
                t = @elapsed begin
                    # copy Cholesky
                    jGP.cK.mat[:] = mainGP.cK.mat
                    jGP.cK.chol.factors[:] = mainGP.cK.chol.factors
                    jGP.alpha[:] = mainGP.alpha
                    jGP.mll = mainGP.mll
                end
                ttotal += t
            elseif (D[i,j] == zero(eltype(D))) || !(mins[j] >= mins[i])
                # solve Cholesky
                t = @elapsed update_mll!(jGP)
                ttotal += t
            elseif D[j,i] == one(eltype(D))
                # j is a sub-region or overlaps
                if mins[j] == mins[i]
                    # easy case
                    # Unfrequent situation
                    #observations_ = @inbounds filter(n -> obsj[obs[i][n]], 1:length(obs[i]))
                    idcs = 1:sum(obsj)
                    t = @elapsed begin
                        jGP.cK.mat[:] = mainGP.cK.mat[idcs,idcs]
                        jGP.cK.chol.factors[:] = mainGP.cK.chol.factors[idcs,idcs]
                        update_mll!(jGP, kern=false, domean=false, noise=false)
                    end
                    ttotal += t
                else
                    # solve with low-rank update
                    # Most frequent situation
                    #observations_ = @inbounds filter(n -> obsj[obs[i][n]], 1:length(obs[i]))
                    #Δ = xor.(obsi, obsj)
                    #deltaobs = @inbounds filter(n -> Δ[obs[i][n]], 1:length(obs[i]))
                    #toupdate = @inbounds filter(n -> any(mainGP.x[:,n] .< mins[j]), deltaobs)

                    toupdate = 1:(1+findfirst(obsj)-findfirst(obsi))

                    # only do low-rank updates if sufficiently stable
                    if (length(toupdate) / sum(obsj)) < τ
                        CC = copy(mainGP.cK.chol.factors)
                        d = size(CC,1)
                        t = @elapsed for n in toupdate
                            AdvancedCholesky.lowrankupdate!(CC,
                                                            view(CC,n,(n+1):d),
                                                            (n+1),
                                                            mainGP.cK.chol.uplo)
                        end
                        ttotal += t
                        t = @elapsed begin
                            CC.factors[observations_,observations_]
                            jGP.cK.mat[:] = mainGP.cK.mat[observations_,observations_]
                            jGP.cK.chol.factors[:] = CC[observations_,observations_]
                            update_mll!(jGP, kern=false, domean=false, noise=false)
                        end
                        ttotal += t
                    else
                        # solve Cholesky
                        t = @elapsed update_mll!(jGP)
                        ttotal += t
                    end
                end
            else
                Σbuffer = GaussianProcesses.mat(jGP.cK)
                GaussianProcesses.cov!(Σbuffer, jGP.kernel, jGP.x, jGP.x, jGP.data)
                noise = eltype(Σbuffer)(exp(2*jGP.logNoise)+eps())
                @inbounds Σbuffer[diagind(Σbuffer)] .+= noise

                F = jGP.cK.chol.factors

                # j isa larger than main region
                if mins[j] == mins[i]
                    # easy case
                    # Unfrequent situation
                    @inbounds F[1:sum(obsi),1:sum(obsi)] = mainGP.cK.chol.factors
                    t = @elapsed AdvancedCholesky.chol_continue(F, sum(obsi)+1)
                    ttotal += t
                    t = @elapsed begin
                        update_mll!(jGP, kern=false, domean=false, noise=false)
                    end
                    ttotal += t
                else
                    # solve with low-rank update & continue computation
                    # Most frequent situation
                    observations_ = @inbounds filter(n -> obsj[obs[i][n]], 1:length(obs[i]))
                    observations2_ = @inbounds filter(n -> obsi[obs[j][n]], 1:length(obs[j]))
                    Δ = xor.(obsi, obsj)
                    deltaobs = @inbounds filter(n -> Δ[obs[i][n]], 1:length(obs[i]))
                    toupdate = @inbounds filter(n -> any(mainGP.x[:,n] .< mins[j]), deltaobs)

                    toupdate = 1:(1+findfirst(obsj)-findfirst(obsi))

                    # only do low-rank updates if sufficiently stable
                    if (length(toupdate) / sum(obsj)) < τ
                        CC = copy(mainGP.cK.chol.factors)
                        d = size(CC,1)
                        t = @elapsed @inbounds for n in toupdate
                            AdvancedCholesky.lowrankupdate!(CC,
                                                            view(CC,n,(n+1):d),
                                                            (n+1),
                                                            mainGP.cK.chol.uplo)
                        end

                        ttotal += t
                        N = sum(obsi) - length(toupdate)
                        @inbounds F[1:N,1:N] = CC[length(toupdate):end,length(toupdate):end]
                        t = @elapsed AdvancedCholesky.chol_continue(F, N+1)
                        ttotal += t
                        t = @elapsed begin
                            update_mll!(jGP, kern=false, domean=false, noise=false)
                        end
                        ttotal += t
                    else
                        # solve Cholesky
                        t = @elapsed update_mll!(jGP)
                        ttotal += t
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
