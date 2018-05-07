function spn_likelihood(node::GPLeaf)
    return node.gp.mll
end

function spn_likelihood(node::FiniteSplitNode)
    return sum(spn_likelihood(child) for child in children(node))
end

function spn_likelihood(node::GPSumNode)
    logw = log.(node.prior_weights)
    return StatsFuns.logsumexp(logw + [spn_likelihood(child) for child in children(node)])
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
        node.Z = StatsFuns.logsumexp(logw_posterior)
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

function spn_logdensityIndep(node::GPLeaf, x::Float64, y::Float64, results::Dict{Int, Float64})
    if !haskey(results, node.id)
        μ, σ = predict_y(node.gp, [x])
        results[node.id] = logpdf(Normal(μ[1], σ[1]), y)
    end
end

function spn_logdensityIndep(node::FiniteSplitNode, x::Float64, y::Float64, results::Dict{Int, Float64})
    if !haskey(results, node.id)
        
        ci = (x <= node.split[1]) ? 1 : 2 
        spn_logdensityIndep(children(node)[ci], x, y, results)
        results[node.id] = results[children(node)[ci].id]
        #results[node.id] = SPNGPResult(μ, μ, σ2)
    end
end

function spn_logdensityIndep(node::GPSumNode, x::Float64, y::Float64, results::Dict{Int, Float64})
    if !haskey(results, node.id)
        
        logw = log.(node.posterior_weights)
        logp = ones(length(node))
        for (ci, child) in enumerate(children(node))
            spn_logdensityIndep(child, x, y, results)
            logp[ci] = results[child.id]
        end     
        
        results[node.id] = StatsFuns.logsumexp(logw + logp)
    end
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
        
        μ = 0.0
        σ2 = 0.0
                
        for child in children(node)
            spn_predictIndep(child, x, results)
            r = results[child.id]
            
            σ2 += 1./r.stdsqr
            μ += r.mean/r.stdsqr
        end
        
        σ2 = 1./σ2
        μ *= σ2
        
        spn_predictIndep(children(node)[ci], x, results)
        results[node.id] = results[children(node)[ci].id]
        #results[node.id] = SPNGPResult(μ, μ, σ2)
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
            
            μ2 += w[ci] * r.mean^2
            σ2 += w[ci] * r.stdsqr
        end     
        
        σ2 = σ2 + μ2 - μ^2
        
        results[node.id] = SPNGPResult(μ, μ, σ2)
    end
end

function spn_predict_moment(node::GPLeaf, x::Vector, obs::Vector, results::Dict{Int, Dict{Int, SPNGPResult}})
    if !haskey(results, node.id)
        μ, σ = predict_y(node.gp, x)
        
        r = Dict(obs[i] => SPNGPResult(μ[i], μ[i]^2, σ[i].^2) for i in 1:length(obs))
        results[node.id] = r
    end
end

function spn_predict_moment(node::FiniteSplitNode, x::Vector, obs::Vector, results::Dict{Int, Dict{Int, SPNGPResult}})
    if !haskey(results, node.id)
        
        c = (x .<= node.split[1])
        
        spn_predict_moment(children(node)[1], x[c], obs[c], results)
        spn_predict_moment(children(node)[2], x[!c], obs[!c], results)
        
        r = Dict{Int, SPNGPResult}()
        for i in 1:length(obs)
            if c[i]
                r[obs[i]] = results[children(node)[1].id][obs[i]]
            else
                r[obs[i]] = results[children(node)[2].id][obs[i]]
            end
        end
        
        results[node.id] = r
    end
end

function spn_predict_moment(node::GPSumNode, x, obs::Vector, results::Dict{Int, Dict{Int, SPNGPResult}})
    
    if !haskey(results, node.id)
        
        for child in children(node)
            spn_predict_moment(child, x, obs, results)
        end
        
        μ = zeros(length(x))
        μ2 = zeros(length(x))
        σ2 = zeros(length(x))
        
        w = node.posterior_weights
        
        for k in 1:length(node)
            
            μsk = [results[children(node)[k].id][xi].mean for xi in obs]
            σ2sk = [results[children(node)[k].id][xi].stdsqr for xi in obs]
            
            μ += μsk .* w[k]
            σ2 += σ2sk .* w[k]
            σ2 += μsk.^2 .* w[k]
        end
        
        σ2 .-= μ.^2
        
        results[node.id] = Dict(obs[i] => SPNGPResult(μ[i], μ[i]^2, σ2[i]) for i in 1:length(obs))
    end
end

# results
mutable struct SPNGParamResult
    loglength::Float64
    haslength::Bool
    logsigma::Float64
    hassigma::Bool
end

function spn_wavelength(node::GPLeaf, x::Float64, r::Dict{Int, SPNGParamResult})
    if !haskey(r, node.id)
        
        pnames = get_param_names(node.gp.k)
        rparam = SPNGParamResult(0.,false, 0., false)
        
        if length(pnames) == 2
            rparam.loglength = get_params(node.gp.k)[findfirst(pnames .== :ll)]
            rparam.haslength = true
            rparam.logsigma = get_params(node.gp.k)[findfirst(pnames .== :lσ)]
            rparam.hassigma = true
        else
            rparam.logsigma = get_params(node.gp.k)[findfirst(pnames .== :ll)]
            rparam.hassigma = true
        #if :ll in pnames
        #    rparam.loglength = get_params(node.gp.k)[findfirst(pnames .== :ll)]
        #    rparam.haslength = true
        #end
        #if :lσ in pnames
        #    rparam.logsigma = get_params(node.gp.k)[findfirst(pnames .== :lσ)]
        #    rparam.hassigma = true
        end
        
        r[node.id] = rparam
    end        
end

function spn_wavelength(node::FiniteSplitNode, x::Float64, r::Dict{Int, SPNGParamResult})
    if !haskey(r, node.id)
        ci = (x <= node.split[1]) ? 1 : 2
        spn_wavelength(children(node)[ci], x, r)
        r[node.id] = r[children(node)[ci].id]
    end
end

function spn_wavelength(node::GPSumNode, x::Float64, r::Dict{Int, SPNGParamResult})
    
    if !haskey(r, node.id)
        for child in children(node)
            spn_wavelength(child, x, r)
        end

        logw = log.(node.posterior_weights)

        rparam = SPNGParamResult(0.,false, 0., false)
        lengths = map(child -> r[child.id].loglength, children(node))
        haslengths = map(child -> r[child.id].haslength, children(node))
        sigmas = map(child -> r[child.id].logsigma, children(node))
        hassigmas = map(child -> r[child.id].hassigma, children(node))

        length = if any(haslengths)
            StatsFuns.logsumexp(lengths[haslengths] + logw[haslengths])
        else
            0
        end

        sigma = if any(hassigmas)
            StatsFuns.logsumexp(sigmas[hassigmas] + logw[hassigmas])
        else
            0
        end

        rparam.loglength = length
        rparam.haslength = any(haslengths)
        rparam.logsigma = sigma
        rparam.hassigma = any(hassigmas)

        r[node.id] = rparam
    end
end

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
        node.posterior = StatsFuns.logsumexp(logw + [spn_posterior(child) for child in children(node)])
        node.posteriorDirty = false
    end
    return node.posterior
end
