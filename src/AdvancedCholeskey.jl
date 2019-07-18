"""
    AdvancedCholeskey module

This module aims to implement extensions to existing Cholesky factorisations
available in Julia. 

Contains:
* `chol_continue!`: Continue a partial Cholesky factorisation.
* `test1`: Test for `chol_continue!`
* `rightlooking_cholesky!`: Right-looking Cholesky factorization (WIP).

"""
module AdvancedCholeskey

using LinearAlgebra
using BenchmarkTools

export genCov, test, rightlooking_cholesky!

# Helper function
genCov(D::Int) = Symmetric(rand(D,D) .+ Matrix(I*D,D,D))

"""
Right-looking blocked Cholesky factorization.

The right-looking Cholesky fact. favors parallelism over data locality
by quickly exposing a large volume of work. At the same time, it modifies
the entire trailing submatrix.

1. POTRF (current triangle)
2. TRSM (sub-matrix below current triangle)
3. SYRK (trailing triangle)

WIP!
"""
function rightlooking_cholesky!(A::Symmetric{T,Array{T,2}}, 
                                start::Int, 
                                stop::Int; 
                                ispotrf::Bool = false) where {T<:LinearAlgebra.BlasFloat}
    e = size(A,1)
    if ispotrf 
        _, info = LAPACK.potrf!('L', view(A.data, start:stop, start:stop))
        @assert info == zero(info)
    end

    if e > stop
        L = tril(A[start:stop, start:stop])
        BLAS.trsm!('L','U','N','N',one(T),L, view(A.data, stop+1:e, start:stop))

    end
    A
end

function test2()
    D = 6
    P = 3

    Σ = genCov(D)
    A = deepcopy(Σ)
    B = deepcopy(Σ)

    LAPACK.potrf!('U', view(A.data, 1:P, 1:P))
    LAPACK.potrf!('U', view(B.data, 1:P, 1:P))

    ki = P+1
    n = D
    t1 = @elapsed begin
    @inbounds begin
        for k = 1:ki-1
            AkkInv = inv(copy(A.data[k,k]'))
            for j in ki:n
                @simd for i = 1:k-1
                    A.data[k,j] -= A.data[i,k]'A.data[i,j]
                end
                A.data[k,j] = AkkInv*A.data[k,j]
            end
        end
    end
    end

    t2 = @elapsed BLAS.trsm!('L','U','T','N',1.0, triu(view(B.data, 1:P,1:P)), view(B.data, 1:P, ki:n))

    return A, B, t1, t2
end

"""
Run a simple test on `chol_continue!`.

Return:
* `t1`: time taken for LinearAlgebra.cholesky!
* `t2`: time taken for LAPACK.potrf! and chol_continue!
* `Δ`: difference between Cholesky factorisations
"""
function test1(;useBLAS=true)
    D = 50
    P = 10
    Σ = genCov(D)
    U = deepcopy(Σ)
    A = deepcopy(Σ)

    t1 = @elapsed LinearAlgebra.cholesky!(U)
    t2 = @elapsed begin
        LAPACK.potrf!('U', view(A.data, 1:P, 1:P))
        chol_continue!(A.data, UpperTriangular, P+1; useBLAS = useBLAS)
    end

    return t1, t2, sum(abs.(UpperTriangular(A) .- UpperTriangular(U)))
end

"""
# Continue a Cholesky decomposition of `A` from `ki` on.
`chol_continue!(A::AbstractMatrix, ::Type{UpperTriangular}, ki::Int)`

## Usage:

```julia
A = genCov(10)
LAPACK.potrf!('U', view(A.data, 1:5, 1:5))
chol_continue!(A.data, UpperTriangular, 5+1)
```

"""
function chol_continue!(A::AbstractMatrix{T}, 
                        ::Type{UpperTriangular}, 
                        ki::Int; 
                        useBLAS::Bool = true
                        ) where {T<:LinearAlgebra.BlasFloat}
    @assert !LinearAlgebra.has_offset_axes(A)
    n = LinearAlgebra.checksquare(A)
    @assert ki <= n

    if !useBLAS
        # Native Julia version
        @inbounds begin
            for k = 1:ki-1
                AkkInv = inv(copy(A[k,k]'))
                for j in ki:n
                    @simd for i = 1:k-1
                        A[k,j] -= A[i,k]'A[i,j]
                    end
                    A[k,j] = AkkInv*A[k,j]
                end
            end
        end
    else
        # BLAS version (faster)
        P = ki-1
        triu!(view(A, 1:P, 1:P))
        BLAS.trsm!('L','U','T','N',one(T), view(A, 1:P, 1:P), view(A, 1:P, ki:n))
    end

    @inbounds begin
        for k = ki:n
            @simd for i = 1:k - 1
                A[k,k] -= A[i,k]'A[i,k]
            end
            Akk, info = LinearAlgebra._chol!(A[k,k], UpperTriangular)
            if info != 0
                return UpperTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(copy(Akk'))
            for j = k + 1:n
                @simd for i = 1:k - 1
                    A[k,j] -= A[i,k]'A[i,j]
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
    end
    return UpperTriangular(A), convert(LinearAlgebra.BlasInt, 0)
end


end # module
