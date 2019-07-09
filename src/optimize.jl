"""
    optimize!(gp::GPBase; kwargs...)
Optimise the hyperparameters of Gaussian process `gp` based on type II maximum likelihood estimation. This function performs gradient based optimisation using the Optim pacakge to which the user is referred to for further details.
# Keyword arguments:
    * `domean::Bool`: Mean function hyperparameters should be optmized
    * `kern::Bool`: Kernel function hyperparameters should be optmized
    * `noise::Bool`: Observation noise hyperparameter should be optimized (GPE only)
    * `lik::Bool`: Likelihood hyperparameters should be optimized (GPMC only)
    * `meanbounds`: [lowerbounds, upperbounds] for the mean hyperparameters
    * `kernbounds`: [lowerbounds, upperbounds] for the kernel hyperparameters
    * `noisebounds`: [lowerbound, upperbound] for the noise hyperparameter
    * `kwargs`: Keyword arguments for the optimize function from the Optim package
"""
function optimizeLocally!(gp::Union{GPSumNode, GPSplitNode}; method = LBFGS(),
                   domean::Bool = true, kern::Bool = true,
                   noise::Bool = true, lik::Bool = true,
                   meanbounds = nothing, kernbounds = nothing,
                   noisebounds = nothing, likbounds = nothing, kwargs...)

end
