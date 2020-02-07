using Distances

export KernelFunction, IsoKernel, ArdKernel
export IsoSE, IsoLinear
export ArdSE, ArdLinear
export getvariance, getlengthscales
export kernelmatrix, kernelmatrix!, kappa
export updategradients!, getgradients
export getdistancematrix

abstract type KernelFunction end
abstract type IsoKernel <: KernelFunction end
abstract type ArdKernel <: KernelFunction end

function kernelmatrix(kernel::KernelFunction, x1::AbstractMatrix, x2::AbstractMatrix)
    P = getdistancematrix(kernel, x1, x2)
    return kernelmatrix(kernel, P)
end

# using pre-computed K
function kernelmatrix!(kernel::IsoKernel, K::AbstractMatrix, P::AbstractMatrix)
    l = getlengthscales(kernel)^2
    map!(kappa(kernel, l), K, P)
    v = getvariance(kernel)
    return lmul!(v,K)
end
@inline kernelmatrix(kernel::KernelFunction, P::AbstractMatrix) = kernelmatrix!(kernel, zero(P), P)

# using pre-computed K

function umap!(f::Function, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    #Threads.@threads
    for (i,j) in zip(eachindex(A),eachindex(B))
        v = f(@inbounds(B[j]))
        @inbounds A[i] += v
    end
end

function kernelmatrix!(kernel::ArdKernel, K::AbstractMatrix{T}, P::AbstractArray{T,3}) where {T<:Real}
    v = getvariance(kernel)
    ls = getlengthscales(kernel).^2
    fill!(K, zero(T))
    for (d,p) in enumerate(eachslice(P, dims=[3]))
        umap!(kappa(kernel, @inbounds(ls[d])), K, p)
    end

    lmul!(v,K)
    return K
end

function kernelmatrix(kernel::KernelFunction, P::AbstractArray{T,3}) where {T}
    return kernelmatrix!(kernel, zeros(T,size(P,1),size(P,2)), P)
end

@inline getdistancematrix(k::KernelFunction, x1) = getdistancematrix(k, x1, x1)

# == SE ISO kernel ==

mutable struct IsoSE{T<:AbstractFloat} <: IsoKernel
    logℓ::T
    logσ::T
    ∂ℓ::T
    ∂σ::T
end

IsoSE(logℓ, logσ) = IsoSE(logℓ, logσ, zero(logℓ), zero(logσ))

@inline getvariance(k::IsoSE; logscale=false) = logscale ? k.logσ : exp(2*k.logσ)
@inline getstd(k::IsoSE) = exp(k.logσ)
function setvariance!(k::IsoSE, v::AbstractFloat)
    k.logσ = v
end
@inline getlengthscales(k::IsoSE; logscale=false) = logscale ? k.logℓ : exp(k.logℓ)
function setlengthscale!(k::IsoSE{T}, l::T) where {T}
    k.logℓ = l
end

@inline rbfkernel(z::T, l::T) where {T<:AbstractFloat} = exp(-0.5*(z/l))

function kappa(k::IsoSE{T}, l::T) where {T<:AbstractFloat}
    return z->rbfkernel(z, l)
end
@inline getdistancematrix(k::IsoSE, x1, x2) = pairwise(SqEuclidean(), x1, x2, dims=1)

function updategradients!(k::IsoSE, precomp::AbstractMatrix, K::AbstractMatrix, P::AbstractMatrix)
    σ = getstd(k)
    l = getlengthscales(k)^2

    # σ * K
    lmul!(σ, K)

    # 0.5 * trace precomp * 2*σ*K)
    k.∂σ = 0.5*tr(precomp * 2*K)

    # 0.5 * trace precomp * σ*K*(P/l^2)
    K.*=P/l
    k.∂ℓ = 0.5*tr(precomp * K)
    return k
end

@inline getgradients(k::IsoSE) = (k.∂σ, k.∂ℓ)
function setgradients!(k::IsoSE, grad)
    ∂σ, ∂ℓ = grad
    k.∂σ = ∂σ
    k.∂ℓ = ∂ℓ
end

# == SE ARD kernel ==
mutable struct ArdSE{T<:AbstractFloat} <: ArdKernel
    logℓ::Vector{T}
    logσ::T
    ∂ℓ::Vector{T}
    ∂σ::T
end

ArdSE(logℓ, logσ) = ArdSE(logℓ, logσ, zero(logℓ), zero(logσ))

@inline getvariance(k::ArdSE; logscale=false) = logscale ? k.logσ : exp(2*k.logσ)
@inline getstd(k::ArdSE) = exp(k.logσ)
function setvariance!(k::ArdSE, v::AbstractFloat)
    k.logσ = v
end
@inline getlengthscales(k::ArdSE; logscale=false) = logscale ? k.logℓ : exp.(k.logℓ)
@inline getlengthscales(k::ArdSE, d::Int) = exp(k.logℓ[d])
function setlengthscale!(k::ArdSE{T}, l::AbstractVector{T}) where {T}
    k.logℓ[:] = l
end
function setlengthscale!(k::ArdSE{T}, l::T) where {T}
    @assert length(k.logℓ) == 1
    k.logℓ[1] = l
end

function kappa(k::ArdSE{T}, l::T) where {T<:AbstractFloat}
    return z->rbfkernel(z,l)
end

function getdistancematrix(k::ArdSE{T}, x1::AbstractMatrix{T}, x2::AbstractMatrix{T}) where {T}
    P = zeros(T, size(x1,1), size(x2,1), length(k.logℓ))
    for d in Base.axes(P,3)
        @inbounds pairwise!(@view(P[:,:,d]), SqEuclidean(), @inbounds(@view(x1[:,d]))', @inbounds(@view(x2[:,d]))', dims=2)
    end
    return P
    #return @inbounds map(d -> pairwise(SqEuclidean(), view(x1,:,d)', view(x2,:,d)', dims=2), 1:length(k.logℓ))
end

function updategradients!(k::ArdSE,
                      precomp::AbstractMatrix{T},
                      K::AbstractMatrix{T},
                      P::AbstractArray{T,3}) where {T}
    σ = getstd(k)
    ls = getlengthscales(k).^2

    # σ * K
    lmul!(σ, K)

    # 0.5 * trace precomp * 2*σ*K)
    k.∂σ = T(0.5)*tr(precomp * 2*K)

    # 0.5 * trace precomp * σ*K*(P/l^2)
    for (d,p) in enumerate(eachslice(P, dims=[3]))
        @inbounds k.∂ℓ[d] = T(0.5)*tr(precomp * K.*(p/ls[d]) )
    end
    return k
end
@inline getgradients(k::ArdSE) = (k.∂σ, k.∂ℓ)
function setgradients!(k::ArdSE, grad)
    ∂σ, ∂ℓ = grad
    k.∂σ = ∂σ
    k.∂ℓ = ∂ℓ
end

# == Linear ISO kernel ==

mutable struct IsoLinear{T<:AbstractFloat} <: IsoKernel
    logℓ::T
    ∂ℓ::T
end

IsoLinear(logℓ) = IsoLinear(logℓ, zero(logℓ))

@inline getvariance(k::IsoLinear; logscale=false) = logscale ? 0.0 : 1.0
@inline getstd(k::IsoLinear) = 1.0
@inline setvariance!(k::IsoLinear, v) = nothing
@inline getlengthscales(k::IsoLinear; logscale=false) = logscale ? k.logℓ : exp(k.logℓ)
function setlengthscale!(k::IsoLinear, l::AbstractFloat)
    k.logℓ = l
end

@inline linearkernel(z::T, l::T) where {T<:AbstractFloat} = z/l
function kappa(k::IsoLinear, l)
    return z -> linearkernel(z, l)
end

@inline getdistancematrix(k::IsoLinear, x1, x2) = x1 * x2'

function updategradients!(k::IsoLinear, precomp::AbstractMatrix, K::AbstractMatrix, P::AbstractMatrix)
    l = getlengthscales(k)
    k.∂ℓ = 0.5*tr(precomp * -2*K)
    return k
end
@inline getgradients(k::IsoLinear) = (zero(typeof(k.∂ℓ)), k.∂ℓ)
function setgradients!(k::IsoLinear, grad)
    _, ∂ℓ = grad
    k.∂ℓ = ∂ℓ
end

# == Linear ARD kernel ==

mutable struct ArdLinear{T<:AbstractFloat} <: ArdKernel
    logℓ::Vector{T}
    ∂ℓ::Vector{T}
end

ArdLinear(logℓ) = ArdLinear(logℓ, zero(logℓ))

@inline getvariance(k::ArdLinear; logscale=false) = logscale ? 0.0 : 1.0
@inline getstd(k::ArdLinear) = 1.0
@inline setvariance!(k::ArdLinear, v) = nothing
@inline getlengthscales(k::ArdLinear; logscale=false) = logscale ? k.logℓ : exp.(k.logℓ)
@inline getlengthscales(k::ArdLinear, d::Int) = exp(k.logℓ[d])
function setlengthscale!(k::ArdLinear{T}, l::T) where {T}
    @assert length(k.logℓ) == 1
    k.logℓ[1] = l
end
function setlengthscale!(k::ArdLinear{T}, l::AbstractVector{T}) where {T}
    k.logℓ[:] = l
end
function kappa(k::ArdLinear, l)
    return z -> linearkernel(z, l)
end

@inline getdistancematrix(k::ArdLinear, x1, x2) = map(d -> x1[:,d] * x2[:,d]', 1:length(k.logℓ))

function updategradients!(k::ArdLinear,
                      precomp::AbstractMatrix{T},
                      K::AbstractMatrix{T},
                      P::AbstractVector{<:AbstractMatrix{T}}) where {T}
    ls = getlengthscales(k)

    @inbounds for d = 1:length(P)
        map!(kappa(k, ls[d]), K, P[d])
        k.∂ℓ[d] = 0.5*tr(precomp * -2*K)
    end

    return k
end
@inline getgradients(k::ArdLinear) = (zero(eltype(k∂ℓ)), k.∂ℓ)
function setgradients!(k::ArdLinear, grad)
    _, ∂ℓ = grad
    k.∂ℓ = ∂ℓ
end
