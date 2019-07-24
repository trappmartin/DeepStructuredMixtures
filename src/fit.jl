export getOverlap, updateK!, shared_cholesky!

@inline function getOverlap(node::GPNode)
    return [Dict(node => (node.observations, vec(maximum(node.dist.x, dims=2))))]
end

@inline function getOverlap(node::GPSplitNode)
    mapreduce(getOverlap, vcat, children(node))
end

function getOverlap(node::GPSumNode)
    r = map(getOverlap, children(node))

    overlap = first(r) # Vector{Dict}

    for k in 2:length(node)
        for j in 1:length(overlap)
            r_ = r[k] # Vector{Dict}

            grp = overlap[j]
            gref = first(grp)
            grpobs = gref[2][1]

            rref = map(first, r_)
            robs = map(i -> any(rj -> haskey(rj,i[1]), overlap) ? Int[] : i[2][1], rref)

            inters = map(i -> intersect(grpobs, i), robs)
            maxi = argmax(map(length, inters))

            for gk in keys(grp)
                overlap[j][gk] = (inters[maxi],grp[gk][2])
            end

            for ik in keys(r_[maxi])
                overlap[j][ik] = (inters[maxi],r_[maxi][ik][2])
            end
        end
    end

    return overlap
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

    C, info = LAPACK.potrf!('U', Σbuffer)
    @assert info == 0

    chol = Cholesky(C, 'U', info)

    gp.cK = PDMat(Σbuffer, chol)
    return chol
end

function shared_cholesky!(spn::Union{GPSumNode, GPSplitNode})

    r = getOverlap(spn)

    selectedNodes = Vector{GPNode}(undef, length(r))
    chols = Vector{Cholesky}(undef, length(r))

    for (j, grp) in enumerate(r)
        ks = collect(keys(grp))
        i = argmax(map(k -> grp[k][2], ks))
        obs = grp[ks[i]][1]

        # compute cholesky factorization
        chol = updateK!(ks[i], obs)
        GaussianProcesses.update_mll!(ks[i].dist, noise = false, kern = false)

        selectedNodes[j] = ks[i]
        chols[j] = chol


        # update all other GPs in the group
        others = setdiff(1:length(ks), i)
        @info others

    end

    selectedNodes, chols
end

