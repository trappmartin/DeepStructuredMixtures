# Deep Mixtures of Gaussian Processes

This package implements deep hierarchical mixtures of Gaussian processes in Julia 1.3.

## Installation
To use this package you need to have Julia 1.3 installed on your machine.

Inside the Julia REPL, you can install the package using:
```julia
using Pkg()
Pkg.install("https://github.com/trappmartin/DeepStructuredMixtures")
```

After the installation you can run the package using:
```julia
using DeepStructuredMixtures
```

## API

A Gaussian Processes object can be instantiate and trained using the following commands:
```julia
# create some training data
x = randn(100, 2) # create 100 samples with 2 dimensions
y = randn(100) # create 100 samples

# select a mean function (only ConstMean is implemented atm)
meanf = ConstMean(0.0)

# select a kernel function (IsoSE or IsoLinear)
kernelf = IsoSE(1.0, 1.0)

# construct a Gaussian process
gp = GaussianProcess(x, y, mean = meanf, kernel = kernelf)

# train a Gaussian process for 1000 iterations using RMSProp
train!(gp, iterations = 1_000)

# make predictions
xtest = randn(50, 2)
m, S = prediction(gp, xtest)

# Note that S is a full covariance matrix, use diag if necessary
s = diag(S)

# plot the Gaussian process
plot(gp)
```

Models for distributed Gaussian process regression can be instantiated and trained as follows:

```julia
x, y ... # some training data

# Number of splits per split node
K = 8

# Number of children under a sum node (>1 allows inference over splits)
V = 2

# Minimum number of observations per Gaussian process
M = 50

# kernel function / functions
kernel = IsoSE(1.0, 1.0) # for a single SE kernel
kernel = ArdSE(ones(10), 1.0) # for a SE kernel with ARD for 10 dimensional data.
kernel = KernelFunction[IsoSE(1.0, 1.0), IsoLinear(1.0)] # for inference over kernels

# mean function
meanf = ConstMean(0.0) # zero mean function

# build a (generalized) product of experts (PoE) model
model1 = buildPoE(x, y, K; generalized = true, M = M, kernel = kernel, meanFun = meanf)

# build a (robust) Bayesian comittee machine (BCM) model with variance prior = 0.1
model2 = buildrBCM(x, y, K, 0.1; robust = true, M = M, kernel = kernel, meanFun = meanf)

# build a deep structured mixture of GPs model
model3 = buildDSMGP(x, y, K, V; M = M, kernel = kernel, meanFun = meanf)

# fit the DSMGP model using shared Cholesky fitting
fit!(model3)

# fit the PoE model using naive fitting
fit_naive!(model1.root)

# train a model using RMSProp
train!(model1)
train!(model2)
train!(model3)

# finetune DSMGP model parameters using RMSProp
finetune!(model3)

# infer the best splits and kernel functions
update!(model3)

# infer only the kernel function
infer!(model3)

# get statistics on the model
stats(model3.root)

# make predictions
xtest ... # some test data
m1, s1 = predict(model1, xtest) # Note that s is only a vector
m2, s2 = predict(model2, xtest) # Note that s is only a vector
m3, s3 = predict(model3, xtest) # Note that s is only a vector

# plot the models
plot(model1, label = "gPoE")
plot!(model2, label = "rBCM")
plot!(model3, label = "SPN-GP")
```
