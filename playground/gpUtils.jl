import GaussianProcesses.mcmc
using GaussianProcesses: get_params, get_param_names, update_target_and_dtarget!
using Optim, PDMats, Distances, FastGaussQuadrature, Klara

function mcmcKernel(gp::GPBase;
              sampler::Klara.MCSampler=Klara.HMC(),
              nIter::Int = 1000,
              burnin::Int = 0,
              thin::Int = 1)
    
    function logpost(hyp::Vector{Float64})  #log-target
        p = get_params(gp)
        p[2:3] = hyp
        set_params!(gp, p)
        return update_target!(gp)
    end

    function dlogpost(hyp::Vector{Float64}) #gradient of the log-target
        Kgrad_buffer = Array{Float64}(gp.nobsv, gp.nobsv)
        p = get_params(gp)
        p[2:3] = hyp
        set_params!(gp, p)
        update_target_and_dtarget!(gp, noise=false, mean=false, kern=true)
        return gp.dtarget
    end
    
    start = get_params(gp)
    starting = Dict(:p=>start[2:3])
    q = BasicContMuvParameter(:p, logtarget=logpost, gradlogtarget=dlogpost) 
    model = likelihood_model(q, false)                               #set-up the model
    job = BasicMCJob(model, sampler, BasicMCRange(nsteps=nIter, thinning=thin, burnin=burnin), starting)   #set-up MCMC job
    print(job)                                             #display MCMC set-up for the user
    
    run(job)                          #Run MCMC
    chain = Klara.output(job)         # Extract chain
    set_params!(gp,start)      #reset the parameters stored in the GP to original values
    return chain.value
end 

function optimize2!(gp::GPBase; method=LBFGS(), 
            mean::Bool=true, kern::Bool=true, noise::Bool=true, lik::Bool=true,
            lowerBound=-10, upperBound=10, iterations = 100, kwargs...)
    
    params_kwargs = GaussianProcesses.get_params_kwargs(typeof(gp); mean=mean, kern=kern, noise=noise, lik=lik)
    
    function ltarget(hyp::Vector{Float64})
        prev = get_params(gp; params_kwargs...)
        try
            set_params!(gp, hyp; params_kwargs...)
            update_target!(gp)
            return -gp.target
        catch err
            # reset parameters to remove any NaNs
            set_params!(gp, prev; params_kwargs...)

            if !all(isfinite.(hyp))
                println(err)
                throw(err)
            elseif isa(err, ArgumentError)
                println(err)
                throw(err)
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                throw(err)
            else
                throw(err)
            end
        end        
    end

    function ltarget_and_dltarget!(grad::Vector{Float64}, hyp::Vector{Float64})
        prev = get_params(gp; params_kwargs...)
        try
            set_params!(gp, hyp; params_kwargs...)
            update_target_and_dtarget!(gp; params_kwargs...)
            grad[:] = -gp.dtarget
            return -gp.target
        catch err
            # reset parameters to remove any NaNs
            set_params!(gp, prev; params_kwargs...)
            if !all(isfinite.(hyp))
                println(err)
                throw(err)
            elseif isa(err, ArgumentError)
                println(err)
                throw(err)
            elseif isa(err, Base.LinAlg.PosDefException)
                println(err)
                throw(err)
            else
                throw(err)
            end
        end
    end

    function dltarget!(grad::Vector{Float64}, hyp::Vector{Float64})
        ltarget_and_dltarget!(grad::Vector{Float64}, hyp::Vector{Float64})
    end
        
    xinit = get_params(gp; params_kwargs...)
    func = Optim.OnceDifferentiable(ltarget, dltarget!, ltarget_and_dltarget!, xinit)
    init = GaussianProcesses.get_params(gp; params_kwargs...)

    min_results = init
    
    try
        results = Optim.optimize(func, init, method, Optim.Options(iterations = iterations))
        min_results = Optim.minimizer(results)
    catch err
        println(err)
        min_results = init
    end
        
       
    #min_results = Optim.minimizer(results)
        
    min_results[min_results .> upperBound] = upperBound
    min_results[min_results .< lowerBound] = lowerBound
    
    GaussianProcesses.set_params!(gp, min_results; params_kwargs...)
    GaussianProcesses.update_target!(gp)
    
    return min_results
end