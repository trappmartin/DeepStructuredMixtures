using Revise
using DeepGaussianProcessExperts
using DelimitedFiles
using Plots, StatsPlots
using StatsFuns: logistic
using GaussianProcesses
using LinearAlgebra, Random, Printf

N = 200

# --                   -- #
# 1D regression problem   #
# --                   -- #
Random.seed!(123)
x = rand(-4:0.0001:4, N, 1) .+ randn(N)
f(x) = 0.5*abs.(x) + sin.(0.5*x) + 0.2*x
y = vec(f(x) + 0.2 * randn(N))

#y = y .- mean(y)

# SPN-GP with a single GP (full GP)
config = SPNGPConfig(
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
c_root = mapreduce(n -> n^3, +, stats(root)[:ndata])
plot(root, title = "GP - Complexity: O($(@sprintf("%.0E", c_root)))")
savefig("1D-GP.png")

# SPN-GP with a mutliple independent GPs
config = SPNGPConfig(
    SE(log(1.0), log(1.0)),
    log(1.0),
    20, # max num.samples
    3, # K
    4, # V
    1, # depth
    0.6,
    true
)
root = buildTree(x, y, config);

@inline getRegions(node::GPNode) = [node.observations]
@inline getRegions(node::GPSplitNode) = mapreduce(getRegions, vcat, children(node))
@inline getRegions(node::GPSumNode) = mapreduce(getRegions, vcat, children(node))
r = getRegions(root)



@inline getOverlap(node::GPNode) = getRegions(node)
@inline getOverlap(node::GPSplitNode) = mapreduce(getOverlap, vcat, children(node))
function getOverlap(node::GPSumNode)
    r = map(getOverlap, children(node))

    overlap = r[1]
    for k in 2:length(node)
        for j in 1:length(overlap)
            i = argmax(map(length, map(r_ -> intersect(overlap[j], r_), r[k])))
            overlap[j] = intersect(overlap[j], r[k][i])
        end
    end

    return overlap
end

r = getOverlap(root)


DeepGaussianProcessExperts.update!(root)
c_root = mapreduce(n -> n^3, +, stats(root)[:ndata])






plot(root, title = "SPN-GP - Complexity: O($(@sprintf("%.0E", c_root)))")
savefig("1D-SPN-GP.png")


# --                   -- #
# 2D regression problem   #
# --                   -- #
Random.seed!(123)
x = rand(-4:0.0001:4, N, 2) .+ randn(N,2)
f(x) = sin.(0.5 * map(n -> norm(x[n,:]), 1:size(x,1))) + (x[:,1] .+ 0.01 * randn()).^2 + 2.6 * sin.(x[:,2])
y = f(x) + 0.05 * randn(N)

#y = y .- mean(y)

xtest = rand(-3.5:0.01:4.5, 50, 2) .+ randn(50,2)
ytest = f(xtest) + 0.05 * randn(50)

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
c_root = mapreduce(n -> n^3, +, stats(root)[:ndata])
plot(root, n=100, seriescolor=:blues, fill=true, lw=0.1,
    var=true, title = "Full GP (variance) - Complexity: O($(@sprintf("%.0E", c_root)))")
savefig("fullGP_var.png")

plot(root, n=100, seriescolor=:blues, fill=true, lw=0.1,
    var=false, title = "Full GP (mean) - Complexity: O($(@sprintf("%.0E", c_root)))")
savefig("fullGP_mean.png")

# SPN-GP with a mutliple independent GPs
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(10.0),
    20,
    3, # K
    6, # V
    3,
    0.5,
    true
)
Random.seed!(1)
root = buildTree(x, y, config);
c_root = mapreduce(n -> n^3, +, stats(root)[:ndata])
DeepGaussianProcessExperts.update!(root)
plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = false, legend=false, title = "SPN-GP (mean) - Complexity: O($(@sprintf("%.0E", c_root)))")
savefig("SPN-GP_mean.png")

plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = true, title = "SPN-GP (variance) - Complexity: O($(@sprintf("%.0E", c_root)))")
savefig("SPN-GP_var.png")

# Optimization
config = SPNGPConfig(
    MeanZero(),
    SE(log(1.0), log(1.0)),
    log(1.0),
    50,
    3, # K
    4, # V
    3,
    0.7,
    true
)
root = buildTree(x, y, config);
root, v = optim!(root, x, y, 500, maxdata = 100);
DeepGaussianProcessExperts.update!(root)
plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = false, legend=false, title = "SPN-GP (mean)")
savefig("SPN-GP-opt_mean.png")

plot(root, n=100, seriescolor=:blues,
    fill=true, lw=0.1, var = true, legend=false, title = "SPN-GP (variance)")
savefig("SPN-GP-opt_var.png")
