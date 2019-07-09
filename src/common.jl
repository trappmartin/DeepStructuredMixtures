function getchild(node::GPSplitNode, x::AbstractArray)
    idx = zeros(Int, size(x,1))
    for n in 1:size(x,1)
        k = 1
        while idx[n] == 0
            split = node.split[k]
            d, s = split
            if x[n,d] <= s
                idx[n] = k
            end
            k += 1
        end
    end
    return idx
end

leftGP(node::GPSumNode) = mapreduce(leftGP, vcat, children(node))
leftGP(node::GPSplitNode) = leftGP(first(children(node)))
leftGP(node::GPNode) = node.dist

rightGP(node::GPSumNode) = mapreduce(rightGP, vcat, children(node))
rightGP(node::GPSplitNode) = rightGP(last(children(node)))
rightGP(node::GPNode) = node.dist

function predict(node::GPNode, x::AbstractVector)
    return GaussianProcesses.predict_f(node.dist, x)
end

function predict(node::GPNode, x::AbstractArray)
    return GaussianProcesses.predict_f(node.dist, x')
end

function predict(node::GPSplitNode, x::AbstractArray)
    idx = getchild(node, x)
    μ = zeros(size(x,1))
    σ² = zeros(size(x,1))
    for (k, c) in enumerate(children(node))
        j = findall(idx .== k)
        m, s = x isa AbstractVector ? predict(c, x[j]) : predict(c, x[j,:])
        μ[j] = m
        σ²[j] = s
    end
    return μ, σ²
end

function predict(node::GPSumNode, x::AbstractArray)
    p = zeros(size(x,1),length(node))
    μ = zeros(size(x,1),length(node))
    σ² = zeros(size(x,1),length(node))
    for (k, c) in enumerate(children(node))
        m, s = predict(c, x)

        μ[:,k] = m
        σ²[:,k] = s

        p[:,k] = logweights(node)[k] .- log.(sqrt.(s))
    end

    i = argmax(p, dims=2)

    return μ[i], σ²[i]
end


function getx(node::GPNode)
    return node.dist.x
end

function getx(node::GPSplitNode)
    return mapreduce(c -> getx(c), hcat, children(node))
end

function gety(node::GPNode)
    return node.dist.y
end

function gety(node::GPSplitNode)
    return mapreduce(c -> gety(c), vcat, children(node))
end

function rand(node::GPNode, x::AbstractArray, n::Int)
    return GaussianProcesses.rand(node.dist, x, n)
end

function rand(node::GPSplitNode, x::AbstractArray, n::Int)
    idx = getchild(node, x)
    yhat = zeros(size(x,1), n)
    for (k, c) in enumerate(children(node))
        j = findall(idx .== k)
        yhat[j,:] .= rand(c, x[j], n)
    end
    return yhat
end

# function update!(node::GPNode)
#     GaussianProcesses.update_target!(node.dist)
#     return node.dist.mll
# end
#
# function update!(node::GPSplitNode)
#     return mapreduce(update!, +, children(node))
# end
#
# function update!(node::GPSumNode)
#     lp = update!.(children(node))
#     lw = log.(rand(Dirichlet(length(node), 1.0)))
#     lw .+= lp
#     z = StatsFuns.logsumexp(lw)
#     node.logweights[:] = lw .- z
#     return z
# end

function stats(node::GPNode; dict::Dict{Symbol,Any} = Dict{Symbol,Any}())
    dict[:gps] = get(dict, :gps, 0) + 1
    if !haskey(dict, :ndata)
        dict[:ndata] = Vector{Int}()
    end
    push!(dict[:ndata], length(node.dist.y))
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
    return dict
end
