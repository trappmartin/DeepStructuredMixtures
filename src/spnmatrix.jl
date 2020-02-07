export SDiagonal
export copyvec

using  LinearAlgebra
import LinearAlgebra: lmul!
import Base.fill!

struct SDiagonal{T<:Real,N,MT<:AbstractArray{T}} <: AbstractArray{T,N}
    Ix :: Vector{Dict{Int,Int}}
    V  :: Vector{MT}
end

function SDiagonal()
    return SDiagonal{Float64,1,Vector{Float64}}(Vector{Dict{Int,Int}}(), Vector{Vector{Float64}}())
end

function SDiagonal(Ix::Vector{Vector{Int}}, V::Vector{MT}) where {T<:Real,N,MT<:AbstractArray{T,N}}
    Ixdict = Vector{Dict{Int,Int}}(undef,length(Ix))
    Vsparse = similar(V)
    for i in Base.axes(Ix,1)
        I = Ix[i]
        @assert size(V[i],1) == length(I)
        Vsparse[i] = N < 2 ? copy(V[i][I .>= i]) : copy(V[i][I .>= i,:])
        II = @view I[I .>= i]
        Ixdict[i] = Dict{Int,Int}(j => k for (k,j) in enumerate(II))
    end
    SDiagonal{T,N+1,MT}(Ixdict, Vsparse)
end

function Base.getindex(SD::SDiagonal{T,2,MT}, I::Vararg{Int,2}) where {T,MT<:AbstractArray}
    i,j = I
    i,j = i>j ? (j,i) : (i,j)
    return haskey(SD.Ix[i],j) ? SD.V[i][SD.Ix[i][j]] : zero(T)
end

function Base.getindex(SD::SDiagonal{T,3,MT}, I::Vararg{Int,3}) where {T,MT<:AbstractArray}
    i,j,k = I
    i,j = i>j ? (j,i) : (i,j)
    return haskey(SD.Ix[i],j) ? SD.V[i][SD.Ix[i][j],k] : zero(T)
end

function Base.setindex!(SD::SDiagonal{T,2,MT}, v::Tv, I::Vararg{Int,2}) where {T,MT<:AbstractVector,Tv<:Number}
    i,j = I
    i,j = i>j ? (j,i) : (i,j)
    if haskey(SD.Ix[i],j)
        SD.V[i][SD.Ix[i][j]] = v
    end
end

function Base.setindex!(SD::SDiagonal{T,3,MT}, v::Tv, I::Vararg{Int,3}) where {T,MT<:AbstractArray,Tv<:Number}
    i,j,k = I
    i,j = i>j ? (j,i) : (i,j)
    SD.V[i][SD.Ix[i][j],k] = v
end

function Base.size(SD::SDiagonal{T,N,MT}) where {T,N,MT<:AbstractArray}
    return (length(SD.Ix), length(SD.Ix), size(SD.V[1])[2:(N-1)]...)
end

function LinearAlgebra.lmul!(a::T, B::SDiagonal{TB,N,MT}) where {T<:Number,TB,N,MT}
    for i in eachindex(B.V)
        @inbounds lmul!(a,@view(B.V[i]))
    end
end

function Base.fill!(A::SDiagonal{T,N,MT}, b::Tb) where {T,N,MT,Tb<:Number}
    for i in eachindex(A.V)
        @inbounds fill!(A.V[i],b)
    end
end

function Base.copy(M::SDiagonal{T,N,MT}) where {T,N,MT}
    V = deepcopy(M.V)
    Ix = deepcopy(M.Ix)
    return SDiagonal{T,N,MT}(V,Ix)
end

function Base.zero(M::SDiagonal{T,N,MT}) where {T,N,MT}
    Ix = deepcopy(M.Ix)
    V = deepcopy(M.V)
    for i in eachindex(V)
        @inbounds fill!(V[i],zero(T))
    end
    return SDiagonal{T,N,MT}(V,Ix)
end

function copyvec(M::SDiagonal{T,2,MT}, dim::Int) where {T,MT}
    return deepcopy(M)
end

function copyvec(M::SDiagonal{T,3,MT}, dim::Int) where {T,N,MT}
    V = map(v -> copy(vec(v[:,dim])), M.V)
    return SDiagonal{T,2,typeof(V[1])}(deepcopy(M.Ix),V)
end
