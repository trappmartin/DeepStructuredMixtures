using Distances
import StatsBase.params

export GaussianProcess
export update_cholesky!, prediction
export mll, ∇mll, ∇mll!
export params, setparams!, nparams, getnoise
export updategradients!

mutable struct Scalar
    value::Float64
end

struct GaussianProcess{TX<:AbstractMatrix{<:AbstractFloat},
                       Ty<:AbstractVector{<:AbstractFloat},
                       Tm<:MeanFunction,
                       Tk<:KernelFunction,
                       Tchol<:AbstractFloat,
                       Ta<:AbstractVector{Tchol},
                       Tdata<:AbstractArray}

    x::TX # inputs
    y::Ty # outputs

    mean::Tm # mean function
    kernel::Tk # kernel function
    logNoise::Scalar
    ∂ϵ::Scalar

    D::Int
    N::Int

    cK::Cholesky{Tchol}
    α::Ta
    P::Tdata

end

@inline getnoise(gp::GaussianProcess; logscale=false) = logscale ? gp.logNoise.value : exp(2*gp.logNoise.value)
@inline function setnoise!(gp::GaussianProcess, noise::AbstractFloat)
    gp.logNoise.value = noise
end

function Base.show(io::IO, ::MIME"text/plain", m::GaussianProcess)
    ℓ = mll(m)
    print(io, "Gaussian process\n noise: ",getnoise(m),"\n kernel: ",m.kernel,"\n mean: ",m.mean,"\n mll:",ℓ)
end
Base.show(io::IO, m::GaussianProcess) = print(io, "GP(",m.kernel,", ",m.mean,")")

function GaussianProcess(x::AbstractMatrix,
                         y::AbstractVector;
                         mean = ConstMean(mean(y)),
                         kernel = IsoSE(0.0, 0.0),
                         logNoise = log(7),
                         run_cholesky = false
                        )
    P = getdistancematrix(kernel, x)
    return GaussianProcess(x, y, mean, kernel, logNoise, P; run_cholesky = run_cholesky)
end

function GaussianProcess(x::AbstractMatrix,
                         y::AbstractVector,
                         mean::MeanFunction,
                         kernel::KernelFunction,
                         logNoise::Float64,
                         P::AbstractArray;
                         run_cholesky = false
                        )
    N,D = size(x)
    cK = Cholesky(zeros(N,N), 'L', 0)
    α = zeros(N)
    yy = similar(y)
    apply_subtract!(mean, y, yy)
    gp = GaussianProcess(x, yy, mean, kernel, Scalar(logNoise), Scalar(0.0), D, N, cK, α, P)

    if run_cholesky
        update_cholesky!(gp)
    end
    return gp
end

function update_cholesky!(gp::GaussianProcess{Tx,Ty,Tm,Tk,Tchol,Ta,Tdata}) where {Tx,Ty,Tm,Tk,Tchol,Ta,Tdata}
    Knn = kernelmatrix(gp.kernel, gp.P)
    return update_cholesky!(gp, Knn)
end

function update_cholesky!(gp::GaussianProcess{Tx,Ty,Tm,Tk,Tchol,Ta,Tdata}, Knn::AbstractMatrix) where {Tx,Ty,Tm,Tk,Tchol,Ta,Tdata}

    # reset factors to kernel matrix
    F = gp.cK.factors
    @inbounds F[:] = Tchol.(Knn)

    # compute noise
    noise = Tchol(getnoise(gp) + ϵ)

    # add noise
    σ = @view F[diagind(F)]
    map!(i -> i+noise, σ, σ)

    # solve cholesky
    LAPACK.potrf!('L', F)

    # update α
    # See Rasmussen and Williams, Algorithm 2.1
    gp.α[:] = gp.cK.L' \ (gp.cK.L \ gp.y)

    return gp
end

function prediction(gp,
                 Knt::AbstractMatrix,
                 Ktt::AbstractMatrix,
                 xtest::AbstractMatrix
                )

    # See Rasmussen and Williams, Algorithm 2.1
    mx = get(gp.mean, size(xtest,1))
    μ = mx + Knt' * gp.α

    V = gp.cK.L \ Knt
    Σ = Ktt - V' * V

    noise = eltype(Σ)(getnoise(gp))

    σ = @view Σ[diagind(Σ)]
    map!(i -> i+noise, σ, σ)

    return μ, Σ
end

function prediction(gp, xtest::AbstractMatrix)

    Knt = kernelmatrix(gp.kernel, gp.x, xtest)
    Ktt = kernelmatrix(gp.kernel, xtest, xtest)

    return prediction(gp, Knt, Ktt, xtest)
end

@inline nparams(gp::GaussianProcess) = map(length, params(gp))

function params(gp::GaussianProcess; logscale = false)
    return (getlengthscales(gp.kernel, logscale=logscale), 
            getvariance(gp.kernel, logscale=logscale), 
            getnoise(gp, logscale=logscale))
end

function setparams!(gp::GaussianProcess, lengthscale, variance, noise::AbstractFloat)
    setlengthscale!(gp.kernel, lengthscale)
    setvariance!(gp.kernel, variance)
    setnoise!(gp, noise)
end

function setparams!(gp::GaussianProcess, hyper::AbstractVector)
    setnoise!(gp, hyper[end])
    setvariance!(gp.kernel, hyper[end-1])
    if length(hyper) == 3
        setlengthscale!(gp.kernel, hyper[1])
    else
        setlengthscale!(gp.kernel, hyper[1:end-2])
    end
end

@inline mll(gp) = - (dot(gp.y, gp.α) + logdet(gp.cK) + log2π * gp.N) / 2

updategradients!(gp::GaussianProcess) = updategradients!(gp, gp.P)
function updategradients!(gp::GaussianProcess, P::AbstractArray)
    K = kernelmatrix(gp.kernel, P)
    return updategradients!(gp, K, P)
end

function updategradients!(gp::GaussianProcess, K::AbstractMatrix, P::AbstractArray )
    T = eltype(K)
    precomp = zeros(T, gp.N, gp.N)
    ααinvcK!(precomp, gp.cK, gp.α)

    gp.∂ϵ.value = getnoise(gp) * tr(precomp)
    updategradients!(gp.kernel, precomp, K, P)
end

function copygradients(source::GaussianProcess, dest::GaussianProcess)
    dest.∂ϵ.value = source.∂ϵ.value
    setgradients!(dest.kernel, getgradients(source.kernel))
end

function ∇mll(gp::GaussianProcess)
    updategradients!(gp)
    grad = zeros(sum(nparams(gp)))
    ∇mll!(gp, grad)
    return grad
end

function ∇mll!(gp::GaussianProcess{Tx,Ty,Tm,Tk,Tchol,Ta,Tdata},
               grad::AbstractVector{Tg}
              ) where {Tx,Ty,Tm,Tk,Tchol,Ta,Tdata,Tg<:AbstractFloat}
    return ∇mll!(gp, grad, gp.P)
end

function ∇mll!(gp::GaussianProcess,
               grad::AbstractVector{Tg},
               P::AbstractArray{T}
              ) where {Tg<:AbstractFloat,T}
    K = kernelmatrix(gp.kernel, P)
    return ∇mll!(gp, grad, K, P)
end

function ∇mll!(gp::GaussianProcess,
               grad::AbstractVector{Tg},
               K::AbstractMatrix{T},
               P::AbstractArray{V}
              ) where {Tg<:AbstractFloat,T,V}

    ∂v, ∂l = getgradients(gp.kernel)
    grad[1:end-1] = vcat(∂l, ∂v)
    grad[end] = gp.∂ϵ.value

    return grad
end

function ααinvcK!(out::AbstractMatrix{T}, cK::Cholesky{T}, α::AbstractVector{T}) where {T}
    o = @view out[diagind(out)]
    map!(i -> i-one(T), o, o)

    ldiv!(cK, out)
    BLAS.ger!(1.0, α, α, out)
    return out
end
