using DeepGaussianProcessExperts
using Plots
using GaussianProcesses
using LinearAlgebra, Random

N = 1000

# --                   -- #
# 1D regression problem   #
# --                   -- #
Random.seed!(123)
x = rand(-4:0.0001:4, N, 1) .+ randn(N)
f(x) = 0.5*abs.(x) + sin.(0.5*x) + 0.2*x
y = vec(f(x) + 0.2 * randn(N))

# SPN-GP with a mutliple independent GPs
config = SPNGPConfig(
    SE(log(1.0), log(1.0)), # kernel function / kernel functions
    log(2), # log σ - Noise
    20, # max number of samples per sub-region
    2, # K = number of splits per split node (not used)
    3, # V = number of children under a sum node
    2, # maxiumum depth of the tree
    0.5, # relative noise used to displace split positions
    true # use sum root
)
spn = buildTree(x, y, config);

gpmapping = getLeafIds(spn);
gpids = collect(keys(gpmapping));

D = zeros(Float64, length(gpids), length(gpids));

# update D
getOverlap(spn, D, gpids);



@info "finished building `spn`"
