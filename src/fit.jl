using Base.Threads
using DeepGaussianProcessExperts.AdvancedCholesk

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
    processed = Vector{Int}()
    queued = Vector{Int}()
    for i in ids
        accept = all(j -> D[i,j] == 0, queued)
        if accept
            push!(queued, i)
        end
        obs[i] = findall(gpmapping[gpids[i]].observations)
    end

    ttotal = @elapsed for i in queued

        mainGP = gpmapping[gpids[i]].dist
        obsi = gpmapping[gpids[i]].observations
        minMain = minimum(mainGP.x, dims=2)
        maxMain = maximum(mainGP.x, dims=2)

        # solve the main GP
        update_mll!(mainGP)

        _ids = sort(ids, by = (j) -> D[i,j], rev=true)

        for j in _ids
            if j ∈ queued
                continue
            end

            if j ∉ processed
                push!(processed, j)
            else
                continue
            end

            jGP = gpmapping[gpids[j]].dist
            obsj = gpmapping[gpids[j]].observations

            if D[i,j] == D[j,i] == one(eltype(D))
                # copy Cholesky
                jGP.cK = mainGP.cK
                jGP.alpha = mainGP.alpha
                jGP.mll = mainGP.mll
                continue
            end

            if D[i,j] == zero(eltype(D))
                # solve Cholesky
                update_mll!(jGP)
                continue
            end

            minJ = minimum(jGP.x, dims=2)
            maxK = maximum(jGP.x, dims=2)

            accept = all(minJ .>= minMain)

            if !accept
                # solve Cholesky (low rank downdates are instable!)
                update_mll!(jGP)
            else
                # copy or update Cholesky
                if D[j,i] .== one(eltype(D))
                    # j is a sub-region or overlaps
                    if all(minJ .== minMain)
                        # easy case
                        # Unfrequent situation
                        observations_ = @inbounds filter(n -> obsj[obs[i][n]], 1:length(obs[i]))
                        jGP.cK = PDMat(
                                       mainGP.cK.mat[observations_,observations_],
                                       Cholesky(
                                                mainGP.cK.chol.factors[observations_,observations_],
                                                mainGP.cK.chol.uplo,
                                                mainGP.cK.chol.info
                                               )
                                      )
                        update_mll!(jGP, kernel=false, mean=false, noise=false)
                    else
                        # solve with low-rank update
                        # Most frequent situation
                        observations_ = @inbounds filter(n -> obsj[obs[i][n]], 1:length(obs[i]))
                        Δ = xor.(obsi, obsj)
                        deltaobs = @inbounds filter(n -> Δ[obs[i][n]], 1:length(obs[i]))
                        toupdate = @inbounds filter(n -> any(mainGP.x[:,n] .< minJ), deltaobs)

                        # only do low-rank updates if sufficiently stable
                        if (length(toupdate) / sum(obsj)) < τ
                            @info "low-rank update"
                            CC = deepcopy(mainGP.cK.chol)
                            D = size(CC,1)
                            @inbounds for n in toupdate
                                lowrankupdate!(CC, view(CC.factors,n,(n+1):D), (n+1))
                            end
                        else
                            # solve Cholesky
                            update_mll!(jGP)
                        end
                        @info "2"
                    end
                else
                    # j isa larger than main region
                    if all(minJ .== minMain)
                        # easy case
                        # Unfrequent situation
                        @info "3"
                    else
                        # solve with low-rank update & continue computation
                        # Most frequent situation
                        @info "4"
                    end
                end
            end
        end
    end

    """
    for id in ids
        #while maxval != zero(maxval)
        gptosolve = gpids[id]
        j = argmax(D[id,:])[1]
        othergp = gpids[j]

        # check if we can skip this
        if othergp ∈ solvedIds
            if D[id,j] == 1.0
                # gptosolve is a sub-region of othergp
                # (completely contained in othergp)

                obs1 = gpmapping[gptosolve].observations
                obs2 = gpmapping[othergp].observations

                N = sum(obs1)
                M = sum(obs2)

                if N == M
                    # both regions are equal
                    # simply copy the cholesky
                    t = @elapsed begin
                        gpmapping[gptosolve].dist.cK = gpmapping[othergp].dist.cK
                        gpmapping[gptosolve].dist.alpha = gpmapping[othergp].dist.alpha
                        gpmapping[gptosolve].dist.mll = gpmapping[othergp].dist.mll
                    end
                    ttotal += t
                else
                    # solve using rank-1 upates ?
                    Δ = xor.(obs1, obs2) .& obs1
                    if sum(Δ) > N
                        # solve GP
                        t = @elapsed update_mll!(gpmapping[gptosolve].dist)
                        ttotal += t
                    else
                        t = @elapsed begin

                            Σbuffer = GaussianProcesses.mat(gpmapping[gptosolve].dist.cK)
                            GaussianProcesses.cov!(Σbuffer,
                                                   gpmapping[gptosolve].dist.kernel,
                                                   gpmapping[gptosolve].dist.x,
                                                   gpmapping[gptosolve].dist.x,
                                                   gpmapping[gptosolve].dist.data)

                            noise = exp(2*gpmapping[gptosolve].dist.logNoise)+eps()
                            @inbounds for i in 1:N
                                Σbuffer[i,i] += noise
                            end
                            U = gpmapping[othergp].dist.cK.chol.U
                            C = zeros(eltype(U), N,N)

                            n = 1
                            m = 1
                            @inbounds for j in findall(obs2)
                                if !obs1[j]
                                    m += 1
                                    continue
                                end

                                if sum(Δ[1:j]) > 0

                                else
                                    C[n,n] = U[m,m]
                                    #gpmapping[gptosolve].dist.cK
                                end
                                m += 1
                                n += 1
                            end
                        end
                        ttotal += t
                        #            @info Δ, length(obs2), length(obs1), length(Δ)
                    end
                end

            else
                # solve GP
                t = @elapsed update_mll!(gpmapping[gptosolve].dist)
                ttotal += t
            end
        else
            t = @elapsed update_mll!(gpmapping[gptosolve].dist)
            #            @info "[fit!] solved GP in: t sec"
            ttotal += t
        end

        push!(solvedIds, gptosolve)
    end
    """

    #    @info "[fit!] finished with $ttotal sec taken for Cholesky decompositions"
    ttotal
end

function fit_naive!(spn::Union{GPSplitNode,GPSumNode})

    gpmapping = getLeafIds(spn)
    gpids = collect(keys(gpmapping))
    K = length(gpids)

    ttotal = @elapsed Threads.@threads for gpid in gpids
        update_mll!(gpmapping[gpid].dist)
    end
    #   @info "[fit_naive!] finished with $ttotal sec taken for Cholesky decompositions"
    ttotal
end
