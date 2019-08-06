using DeepGaussianProcessExperts
using Plots
using GaussianProcesses
using LinearAlgebra, Random

N = 400

# --                   -- #
# 1D regression problem   #
# --                   -- #
Random.seed!(123)
x = rand(-4:0.0001:4, N, 1) .+ randn(N)
x = sort(x, dims=1)
f(x) = 0.5*abs.(x) + sin.(0.5*x) + 0.2*x
y = vec(f(x) + 0.2 * randn(N))

t_naive = zeros(4)
t_lr_0 = zeros(4)
t_lr_1 = zeros(4)
t_lr_2 = zeros(4)

#for (i, V) in enumerate([2, 4, 6, 8])
V = 4

    # SPN-GP with a mutliple independent GPs
    config = SPNGPConfig(
        SE(log(1.0), log(1.0)), # kernel function / kernel functions
        log(2), # log σ - Noise
        20, # max number of samples per sub-region
        2, # K = number of splits per split node (not used)
        V, # V = number of children under a sum node
        2, # maxiumum depth of the tree
        0.25, # relative noise used to displace split positions
        true # use sum root
    )
    spn = buildTree(x, y, config);

    gpmapping = getLeafIds(spn);
    gpids = collect(keys(gpmapping));

    D = zeros(Float64, length(gpids), length(gpids));

    # update D
    getOverlap(spn, D, gpids);

#    t_naive[i] = mean(_ -> fit_naive!(spn), 1:100)
#    t_lr_0[i] = mean(_ -> fit!(spn, D, τ = 0.0), 1:100)
#    t_lr_1[i] = mean(_ -> fit!(spn, D, τ = 0.1), 1:100)
    #t_lr_2[i] = mean(_ -> fit!(spn, D, τ = 0.2), 1:100)

#end

#plot([2,4,6,8], t_naive, label = "naive", xlabel = "# children under sum", ylabel = "time taken in sec")
#plot!([2,4,6,8], t_lr_0, label = "tau = 0.0")
#plot!([2,4,6,8], t_lr_1, label = "tau = 0.1")
#plot!([2,4,6,8], t_lr_2, label = "tau = 0.2")

#savefig("timetaken.png")

@info "finished experiment"
