using GaussianProcesses, Plots

# load plotting lib
pgfplots(size = (1200, 1200))

# read in some test data
dataPath = "data/clean/motor.csv"
(data, header) = readcsv(dataPath, header = true)

# make convinence dict
headerDict = Dict(col[2] => col[1] for col in enumerate(header))

# construct x and Y
X = convert(Vector, data[:,headerDict["times"]])
Y = convert(Vector, data[:,headerDict["accel"]])

N = length(X)

# normalize
#X /= maximum(X)
Y /= maximum(Y)

scatter(X, Y)
savefig("plots/motor.png")

# define a GP
meanFun = MeanZero()
kernelLengthScale = log(1.) # log of the inverse length scale
kernelSigma = log(1.) # log of the signal standard deviation
kern = SE(kernelLengthScale, kernelSigma) # squared exponential
obsNoise = log(0.05)

# create a full GP
gp = GP(reshape(X, 1, N), Y, meanFun, kern, obsNoise)

# plot the posterior of a full GP
plot(gp, title="GP with SE kernel (lengthscale: $(kernelLengthScale), sigma: $(kernelSigma)), noise sigma $(obsNoise) and marginal LL: $(gp.target)")
savefig("plots/motor_fullGP.png")

# optimize a full GP
optimize!(gp; method=Optim.BFGS())

# plot posterior of otimized full GP
plot(gp)
savefig("plots/motor_fullGP_opt.png")
