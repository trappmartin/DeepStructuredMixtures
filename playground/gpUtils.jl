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