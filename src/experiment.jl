using Revise
using DeepGaussianProcessExperts
using DelimitedFiles
using Plots, StatsPlots
using StatsFuns: logistic
using GaussianProcesses
using LinearAlgebra, Random

N = 200

# --                   -- #
# 1D regression problem   #
# --                   -- #
Random.seed!(123)
x = rand(-4:0.0001:4, N, 1) .+ randn(N)
f(x) = 0.5*abs.(x) + sin.(0.5*x) + 0.2*x
y = vec(f(x) + 0.2 * randn(N))

scatter(x, y)

# SPN-GP with a single GP (full GP)
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(1.0),
    500,
    2,
    2,
    1,
    0.0,
    false
)
root = buildTree(x, y, config);
plot(root, title = "GP")
savefig("1D-GP.png")

# SPN-GP with a mutliple independent GPs
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(1.0),
    20,
    3, # K
    4, # V
    3,
    0.7,
    true
)
root = buildTree(x, y, config);
plot(root, title = "SPN-GP")
savefig("1D-SPN-GP.png")

# --                   -- #
# 2D regression problem   #
# --                   -- #
Random.seed!(123)
x = rand(-4:0.0001:4, N, 2) .+ randn(N,2)
f(x) = sin.(0.5 * map(n -> norm(x[n,:]), 1:size(x,1))) + (x[:,1] .+ 0.01 * randn()).^2 + 2.6 * sin.(x[:,2])
y = f(x) + 0.05 * randn(N)

xtest = rand(-3.5:0.01:4.5, 50, 2) .+ randn(50,2)
ytest = f(xtest) + 0.05 * randn(50)

# SPN-GP with a single GP (full GP)
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(10.0),
    500,
    2,
    2,
    1,
    0.0,
    false
)
root = buildTree(x, y, config);

plot(root, n=100, seriescolor=:blues, fill=true, lw=0.1,
    var=true, title = "Full GP (variance)")
savefig("fullGP_var.png")

plot(root, n=100, seriescolor=:blues, fill=true, lw=0.1,
    var=false, title = "Full GP (mean)")
savefig("fullGP_mean.png")

yhat, _ = predict(root,xtest)
rmse = sqrt(mean((ytest - yhat).^2))

# SPN-GP with a mutliple independent GPs
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(10.0),
    100,
    3, # K
    4, # V
    3,
    0.7,
    true
)
root = buildTree(x, y, config);

yhat, _ = predict(root,xtest)
rmse = sqrt(mean((ytest - yhat).^2))



DeepGaussianProcessExperts.update!(root)
plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = false, legend=false, title = "SPN-GP (mean)")
savefig("SPN-GP_mean.png")

plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = true, title = "SPN-GP (variance)")
savefig("SPN-GP_var.png")

# Optimization
root, v = optim!(root, x, y, 100, maxdata = 100);

plot(v)
savefig("target.png")

plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = false, legend=false, title = "SPN-GP (mean)")
savefig("SPN-GP-opt_mean.png")

plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, legend=false, title = "SPN-GP (variance)")
savefig("SPN-GP-opt_var.png")
