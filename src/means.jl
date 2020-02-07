export MeanFunction, ConstMean

# Mean functions for a GP

abstract type MeanFunction end

struct ConstMean{T<:AbstractFloat} <: MeanFunction
    m::T
end

function apply_subtract!(m::ConstMean, y::AbstractVector, yout::AbstractVector)
    map!(i -> i - m.m, yout, y)
    return yout
end

function get(m::ConstMean{T}, N::Int) where {T}
    return ones(T,N)*m.m
end
