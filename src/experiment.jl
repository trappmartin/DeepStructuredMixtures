using Revise
using DeepGaussianProcessExperts
using DelimitedFiles
using Plots
using StatsFuns: logistic
using GaussianProcesses
using LinearAlgebra

# make 2D regression problem
N = 200
x = rand(-4:0.0001:4, N, 2) .+ randn(N,2)
f(x) = sin.(0.5 * map(n -> norm(x[n,:]), 1:size(x,1))) + (x[:,1] .+ 0.01 * randn()).^2 + 2.6 * sin.(x[:,2])
y = f(x) + 0.05 * randn(N)

xtest = rand(-3.5:0.01:4.5, 50, 2) .+ randn(50,2)
ytest = f(xtest) + 0.05 * randn(50)

# SPN-GP with a single GP (full GP)
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(0.5),
    500,
    2,
    2,
    1,
    0.0
)
root = buildTree(x, y, config);
#optimize!(root)
plot(root, n=100, seriescolor=:blues, fill=true, lw=0.1,
    var=true, legend = false, title = "Full GP (variance)")
savefig("fullGP_var.png")

plot(root, n=100, seriescolor=:blues, fill=true, lw=0.1,
    var=false, legend = false, title = "Full GP (mean)")
savefig("fullGP_mean.png")


μ, σ² = predict(root, xtest)
sqrt(mean((μ .- ytest).^2))

# SPN-GP with a mutliple independent GPs
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(0.5),
    30,
    3,
    2,
    3,
    0.5
)
root = buildTree(x, y, config);

plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = false, legend=false, title = "SPN-GP (mean)")
savefig("SPN-GP_mean.png")

plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = true, legend=false, title = "SPN-GP (variance)")
savefig("SPN-GP_var.png")

using StatsPlots, StatsFuns

DeepGaussianProcessExperts.target(root)


# update splits


function t_log(t_0, t_n, i, n)
    return t_0 - i ^ ( log(t_0 - t_n) / log(n) )
end

function optim!(root, x, y, n; t0 = 0.7, tn = 0.1)

    v = zeros(n+1)
    v[1] = DeepGaussianProcessExperts.target(root)

    for i in 1:n

        t = t_log(t0, tn, i, n)

        bestConfig = Dict{Symbol,Float64}()
        score = -Inf
        for r in 1:100
            dict = Dict{Symbol,Float64}()
            s = DeepGaussianProcessExperts.resample!(root, x, y, t, dict)
            if isfinite(s)
                if (s > score)
                    score = s
                    bestConfig = dict
                end
            end
        end

        @info i, score
        DeepGaussianProcessExperts.updateStructure!(root, x, y, bestConfig)
        v[i+1] = score
    end

    (root, v)
end

root, v = optim!(root, x, y, 10);

plot(v)

bestConfig = argmax(scores)

vals[]


μ, σ² = predict(root, xtest)
sqrt(mean((μ .- ytest).^2))

spnstats = stats(root)
mean(spnstats[:ndata])
histogram(spnstats[:ndata])


N^3 - spnstats[:gps] * mean(spnstats[:ndata])^3

stats(root)

rx = -5.0:0.5:70.0
yhat = mapreduce(xi -> DeepGaussianProcessExperts.rand(root, [xi], 1), hcat, rx)
scatter(x, y)
plot!(rx, vec(yhat))

# define a GP
meanFun = MeanZero()
kernelLengthScale = log(1.) # log of the inverse length scale
kernelSigma = log(1.) # log of the signal standard deviation
kern = SE(kernelLengthScale, kernelSigma) # squared exponential
obsNoise = log(0.5)

# create a full GP
x = Z[:,1:2]
gp = GP(x', y, meanFun, kern, obsNoise)

optimize!(gp)
plot!(gp, seriescolor=:reds)

# plot the posterior of a full GP
plot(gp, title="GP with SE kernel (lengthscale: $(kernelLengthScale), sigma: $(kernelSigma)), noise sigma $(obsNoise) and marginal LL: $(gp.target)")
savefig("plots/motor_fullGP.png")

# optimize a full GP
optimize!(gp; method=Optim.BFGS())

# plot posterior of otimized full GP
plot(gp)
savefig("plots/motor_fullGP_opt.png")
