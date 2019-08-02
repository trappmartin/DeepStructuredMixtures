export getOverlap, updateK!, fit!, fit_naive!
export getLeafIds

@inline getLeafIds(node::GPNode) = Dict{Symbol,GPNode}(node.id => node)
@inline getLeafIds(node::GPSplitNode) = mapreduce(getLeafIds, merge, children(node))
@inline getLeafIds(node::GPSumNode) = mapreduce(getLeafIds, merge, children(node))

@inline getOverlap(node::GPNode, D, ids) = Dict(findfirst(node.id .== ids) => node)
@inline function getOverlap(node::GPSplitNode, D::Array, ids)
    return mapreduce(c -> getOverlap(c, D, ids), merge, children(node))
end
function getOverlap(node::GPSumNode, D::Array, ids)
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
                        D[n, m] = 1 - (Δn / sum(nnode.observations))
                        D[m, n] = 1 - (Δm / sum(mnode.observations))
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

function fit!(spn::Union{GPSumNode, GPSplitNode}, D::Matrix)

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

                if sum(obs1) == sum(obs2)
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
                    if sum(Δ) > sum(obs1)
                        # solve GP
                        t = @elapsed update_mll!(gpmapping[gptosolve].dist)
                        ttotal += t
                    else
                        t = @elapsed begin

                            i = 1
                            #@inbounds for j in 1:length(obs2)
                            #    if any(j .> Δ)

                            #    else
                                #gpmapping[gptosolve].dist.cK
                            #    end
                            #end
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
#            @info "[fit!] solved GP in: $t sec"
            ttotal += t
        end

        push!(solvedIds, gptosolve)
    end

    #    @info "[fit!] finished with $ttotal sec taken for Cholesky decompositions"
    ttotal
end

function fit_naive!(spn::Union{GPSplitNode,GPSumNode})

    # total time taken for Choleskys
    ttotal = 0

    gpmapping = getLeafIds(spn)
    gpids = collect(keys(gpmapping))
    K = length(gpids)

    for gpid in gpids
 #       @info "[fit_naive!] fitting $gpid"
        t = @elapsed update_mll!(gpmapping[gpid].dist)
        ttotal += t
    end
 #   @info "[fit_naive!] finished with $ttotal sec taken for Cholesky decompositions"
    ttotal
end
