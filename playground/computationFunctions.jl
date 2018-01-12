function spn_likelihood(node::GPLeaf)
    return node.gp.mll
end

function spn_likelihood(node::FiniteSplitNode)
    return sum(spn_likelihood(child) for child in children(node))
end

function spn_likelihood(node::GPSumNode)
    logw = log.(node.prior_weights)
    return logsumexp(logw + [spn_likelihood(child) for child in children(node)])
end

function spn_update!(node::GPLeaf)
    return spn_likelihood(node)
end

function spn_update!(node::FiniteSplitNode)
    return sum(spn_update!(child) for child in children(node))
end

function spn_update!(node::GPSumNode)
    
    if node.zDirty
        logw_prior = log.(node.prior_weights)
        logw_posterior = logw_prior + [spn_update!(child) for child in children(node)]
        node.Z = logsumexp(logw_posterior)
        node.posterior_weights[:] = exp.(logw_posterior - node.Z)
        node.zDirty = false
    end
        
    return node.Z
end

function dirty!(node::FiniteSplitNode)   
    if !node.posteriorDirty
        node.posteriorDirty = true
        for child in children(node)
            dirty!(child) 
        end
    end
end

function dirty!(node::GPSumNode)
    if !(node.posteriorDirty & node.zDirty)
        node.zDirty = true
        node.posteriorDirty = true
        for child in children(node)
            dirty!(child) 
        end
    end
end

function dirty!(node::GPLeaf)
end

# results
struct SPNGPResult
    mean::Float64
    meansqr::Float64
    stdsqr::Float64
end

function spn_predictIndep(node::GPLeaf, x::Float64, results::Dict{Int, SPNGPResult})
    if !haskey(results, node.id)
        μ, σ = predict_y(node.gp, [x])
        r = SPNGPResult(μ[1], μ[1]^2, σ[1].^2)
        results[node.id] = r
    end
end

function spn_predictIndep(node::FiniteSplitNode, x::Float64, results::Dict{Int, SPNGPResult})
    if !haskey(results, node.id)
        
        ci = (x <= node.split[1]) ? 1 : 2
        
        spn_predictIndep(children(node)[ci], x, results)
        results[node.id] = results[children(node)[ci].id]
    end
end

function spn_predictIndep(node::GPSumNode, x::Float64, results::Dict{Int, SPNGPResult})
    
    if !haskey(results, node.id)
        
        μ = 0.0
        μ2 = 0.0
        σ2 = 0.0
        
        w = node.posterior_weights
        
        for (ci, child) in enumerate(children(node))
            spn_predictIndep(child, x, results)
            r = results[child.id]
            μ += w[ci] * r.mean
            μ2 += w[ci] * r.meansqr
            σ2 += w[ci] * r.stdsqr
            #push!(σs, r.stds...)
            #push!(ws, logw[ci] + r.logweights...)
        end
        
        results[node.id] = SPNGPResult(μ, μ2, σ2)
    end
end

#function spn_predict(node::FiniteSplitNode, x::Vector, results::Dict{Int, SPNGPResult})
#    
#    s = x .<= node.split
#    
#    μ = zeros(length(x))
#    σ2 = zeros(length(x))
#    
#    pred1 = spn_predict(children(node)[1], x[s])
#    pred2 = spn_predict(children(node)[2], x[.!s])
#    
#    μ[s] = pred1[1]
#    μ[.!s] = pred2[1]
#    
#    σ2[s] = pred1[2]
#    σ2[.!s] = pred2[2]
#    
#    return (μ, σ2)
#end

#function spn_predict(node::GPSumNode, x)
#    childPredictions = [spn_predict(child, x) for child in children(node)]
#    
#    μs = [pred[1] for pred in childPredictions]
#    σ2s = [pred[2] for pred in childPredictions]
#    
#    μ = zeros(length(x))
#    σ2 = zeros(length(x))
#    
#    for k in 1:length(node)
#       μ += μs[k] .* node.posterior_weights[k]
#       σ2 += σ2s[k] .* node.posterior_weights[k]
#       σ2 += μs[k].^2 .* node.posterior_weights[k]
#    end
#    
#    σ2 .-= μ.^2
#    
#    return (μ, σ2)
#end

function spn_rand(node::GPLeaf, x)
    return rand(node.gp, x)
end

function spn_rand(node::FiniteSplitNode, x)
    s = x .<= node.split
    
    y = zeros(length(x))
    
    y[s] = spn_rand(children(node)[1], x[s])
    y[.!s] = spn_rand(children(node)[2], x[.!s])
    
    return y
end

function spn_rand(node::GPSumNode, x)
    s = rand(Categorical(node.posterior_weights))
    return spn_rand(children(node)[s], x)
end

function spn_randIndep(node::GPLeaf, x)
    return map(xx -> rand(node.gp, [xx])[1], x)
end

function spn_randIndep(node::FiniteSplitNode, x)
    s = x .<= node.split
    
    y = zeros(length(x))
    
    y[s] = spn_randIndep(children(node)[1], x[s])
    y[.!s] = spn_randIndep(children(node)[2], x[.!s])
    
    return y
end

function spn_randIndep(node::GPSumNode, x)
    s = rand(Categorical(node.posterior_weights))
    return spn_randIndep(children(node)[s], x)
end

function spn_posterior(node::GPLeaf)
    return node.gp.target
end

function spn_posterior(node::FiniteSplitNode)
    if node.posteriorDirty
        node.posterior = sum(spn_posterior(child) for child in children(node))
        node.posteriorDirty = false
    end
    return node.posterior
end

function spn_posterior(node::GPSumNode)
    if node.posteriorDirty
        logw = log.(node.posterior_weights)
        node.posterior = logsumexp(logw + [spn_posterior(child) for child in children(node)])
        node.posteriorDirty = false
    end
    return node.posterior
end
