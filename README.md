# Deep Mixtures of Gaussian Processes

This package implements Deep Structured Mixtures of Gaussian Processes (DSMGP) [1] in Julia 1.3.

## Installation
To use this package you need Julia 1.3 installed on your machine.

Inside the Julia REPL, you can install the package in the Pkg mode (type `]` in the REPL):
```julia
pkg> add https://github.com/trappmartin/DeepStructuredMixtures
```

After the installation you can load the package using:
```julia
using DeepStructuredMixtures
```

Note that the package will be compiled the first time you load it.

## Python bridge
The package can be used from python using the excelent pyjulia package: https://github.com/JuliaPy/pyjulia

## Usage
The following example explains the usage of `DeepStructuredMixtures`. Note that this example assume that you have `Plots` installed in your Julia environment.

First, we load the necessary libraries:
```julia
using Plots
using DeepStructuredMixtures
using Random
```

Now we can create some synthetic data or load some real data:
```julia
xtrain = collect(range(0, stop=1, length = 100))
ytrain = sin.(xtrain*4*pi + randn(100)*0.2)
```

We will now use a squared exponential kernel-function with a constant mean-function to fit the DSMGP. See API for more options.
```julia
kernelf = IsoSE(1.0, 1.0)
meanf = ConstMean(mean(xtrain))
```

Now we can construct a DSMGP on our data and find optimial hyperparameters.
```julia
K = 4 # Number of splits per product node
V = 3 # Number of children per sum node
M = 10 # Minimum number of observations per expert

model = buildDSMGP(reshape(xtrain,:,1), ytrain, V, K; M = M, kernel = kernelf, meanFun = meanf)
train!(model)
```

Note that for large data sets it is recommended to train the DSMGP with `V = 1` and use the hyper-parameters to initialise the training of a model with `V > 1`:
```julia
model1 = buildDSMGP(reshape(xtrain,:,1), ytrain, 1, V; M = M, kernel = kernelf, meanFun = meanf)
train!(model1)

# get hyper-parameters
hyp = reduce(vcat, params(leftGP(model1.root), logscale=true))

model = buildDSMGP(reshape(xtrain,:,1), ytrain, K, V; M = M, kernel = kernelf, meanFun = meanf)

# set hyper-parameters instead of learning from scratch
setparams!(model.root, hyp)
train!(model, randinit = false)
```

Finally, we can plot the model:
```julia
plot(model)
```

and use it for predictions:
```julia
xtest = collect(range(0.5, stop=1.5, length = 100))
m, s = predict(model, reshape(xtest,:,1))

err = DeepStructuredMixtures.invΦ((1+0.95)/2)*sqrt.(s)

plot(xtest, m)
plot!(xtest, m + err, primary=false, linestyle=:dot)
plot!(xtest, m - err, primary=false, linestyle=:dot)
```

Note that all methods assume that `xtrain` and `xtest` are matrices, which is why we use `reshape(xtest,:,1)` to reshape the respective vectors to a matrix.


## API

#### Mean functions
```julia
# A constant mean of zero aka zero-mean function.
ConstMean(0.0) 
```

#### Kernel functions
```julia
# A squared exponential kernel-function with lengthscale 1 and std of 1.
IsoSE(1.0, 1.0)

# A squared exponential kernel-function with ARD and lengthscales of 1 and std of 1.
ArdSE(ones(10), 1.0)

# A linear kernel-function with lengthscale of 1.
IsoLinear(1.0)

# A linear kernel-function with ARD and lengthscales of 1.
ArdLinear(ones(10))

# Composition of kernel-function for inference over kernel-functions.
KernelFunction[IsoSE(1.0, 1.0), IsoLinear(1.0)] 
```

#### Models
```julia
# An exact Gaussian process
GaussianProcess(trainx, trainy, mean = meanf, kernel = kernelf)

# A (generalized) product of experts (PoE) model with K splits per node and a miminum of M observations per expert
buildPoE(trainx, trainy, K; generalized = true, M = M, kernel = kernelf, meanFun = meanf)

# A (robust) Bayesian comittee machine (BCM) model with K splits per node and a miminum of M observations per expert
# ! Training not implemented !
buildrBCM(x, y, K; M = M, kernel = kernelf, meanFun = meanf)

# A deep structured mixture of GPs (DSMGP) model with K splits per product node, V children per sum node and a miminum of M observations per expert.
buildDSMGP(x, y, V, K; M = M, kernel = kernelf, meanFun = meanf)
```

#### Training
```julia
# train a model for 1000 iterations using RMSProp
train!(model, iterations = 1_000)

# fit the posterior of a hierarchical model, e.g. gPoE
fit_naive!(model.root)

# fit the posterior of a DSMGP using shared Cholesky
fit!(model)
```

#### Prediction
```julia
# make predictions using a model, i.e., compute mean (s) and variance (s).
m, s = prediction(model, testx)

# plot a model and the training data.
plot(model)
```

## Reference
[1] Martin Trapp, Robert Peharz, Franz Pernkopf and Carl Edward Rasmussen: Deep Structured Mixtures of Gaussian Processes. To appear at the International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.
