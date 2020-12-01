using DeepStructuredMixtures
using Random, LinearAlgebra
using Plots, StatsFuns

# create some random data
D = 2
N = 50

Random.seed!(123)
x = rand(N, D) 
y = cos.(6*(x[:,1] + x[:,2])) - 2*sin.(6*(x[:,1] + x[:,2]))

# plot the data
scatter(x[:,1], x[:,2], zcolor = y)

# learn a DSMGP
K = 2
V = 4
kernelf = IsoSE(log(0.1), log(0.1))
Random.seed!(123)

dsmgp = buildDSMGP(x, y, K, V; ϵ = 0.0, M = 100, kernel = kernelf, logNoise = log(0.1))
update!(dsmgp)

xrange = 0:0.05:1
yrange = xrange

f(x, y) = predict(dsmgp, [x y])[1] 
μ = mapreduce(yy -> mapreduce(xx -> f(xx, yy), hcat, xrange), vcat, yrange)
contour(xrange, yrange, μ, fill=true)
scatter!(x[:,1], x[:,2], zcolor = y)
savefig("fullGP.png")

Random.seed!(123)
dsmgp = buildDSMGP(x, y, K, V; ϵ = 0.1, M = 5, kernel = kernelf, logNoise = log(0.1))
update!(dsmgp)

f(x, y) = predict(dsmgp, [x y])[1] 
μ = mapreduce(yy -> mapreduce(xx -> f(xx, yy), hcat, xrange), vcat, yrange)
contour(xrange, yrange, μ, fill=true)
scatter!(x[:,1], x[:,2], zcolor = y)
savefig("DSMGP.png")

# partition the axis
function replaceGP!(n, d, x, y)
    replaceGP!.(children(n), d, Ref(x), Ref(y))
end

function replaceGP!(n::GPSplitNode, d, x, y)
    di, s = first(n.split)
    if d != di
        replaceGP!.(children(n), d, Ref(x), Ref(y))
    else
        for k in 1:length(n)-1
            deleteat!(n.children, 1)
            deleteat!(n.split, 1)
        end
        replaceGP!(children(n)[1], d, x, y)
    end
end

function replaceGP!(n::GPNode, d, x, y)
    N,D = size(x)
    dims = filter(di -> di != d, 1:D)

    f(xi, lb, ub) = (xi > lb) & (xi <= ub)
    b = map(i -> all(f(x[i,di], n.lb[di], n.ub[di]) for di in dims), 1:N)
    idx = findall(b)
    
    # create a full GP
    GaussianProcess(x[idx,:], y[idx]; 
        kernel = deepcopy(n.dist.kernel),
        mean = deepcopy(n.dist.mean),
        logNoise = n.dist.logNoise.value,
        run_cholesky = true
        )

    return length(idx) 
end

Random.seed!(123)
dsmgp = buildDSMGP(x, y, K, V; ϵ = 0.1, M = 5, kernel = kernelf, logNoise = log(0.1))

replaceGP!(dsmgp.root.children[1], 1, x, y)
replaceGP!(dsmgp.root.children[2], 2, x, y)
reset_weights!(dsmgp)
update!(dsmgp)

f(x, y) = predict(dsmgp, [x y])[1]
μ = mapreduce(yy -> mapreduce(xx -> f(xx, yy), hcat, xrange), vcat, yrange)

contour(xrange, yrange, μ, fill=true)
scatter!(x[:,1], x[:,2], zcolor = y)
savefig("DSMGP_sliced.png")