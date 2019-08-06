"""
# AdvancedCholesky module

This module aims to implement extensions to existing Cholesky factorisations
available in Julia.

"""
module AdvancedCholesky
using LinearAlgebra, Statistics, Random
import LinearAlgebra.lowrankupdate

genCov(D::Int) = Symmetric(rand(D,D) .+ Matrix(I*D,D,D))


function lowrankupdate!(C::Cholesky, v::StridedVector, k::Int)
    lowrankupdate!(C.factors, v, k, C.uplo)
    return C
end

function lowrankupdate!(A::Matrix, v::StridedVector, k::Int, uplo::Char)
    @assert k > 0

    n = length(v)
    if (size(A,1)-(k-1)) != n
        throw(DimensionMismatch("updating vector must fit size of factorization"))
    end
    if uplo == 'U'
        conj!(v)
    end

    for i = k:n

        # Compute Givens rotation
        @inbounds c, s, r = LinearAlgebra.givensAlgorithm(A[i,i], v[i-(k-1)])

        # Store new diagonal element
        @inbounds A[i,i] = r

        # Update remaining elements in row/column
        if uplo == 'U'
            @inbounds for j = i + 1:n
                Aij = A[i,j]
                vj  = v[j-(k-1)]
                A[i,j]  =   c*Aij + s*vj
                v[j-(k-1)]    = -s'*Aij + c*vj
            end
        else
            @inbounds for j = i + 1:n
                Aji = A[j,i]
                vj  = v[j-(k-1)]
                A[j,i]  =   c*Aji + s*vj
                v[j-(k-1)]    = -s'*Aji + c*vj
            end
        end
    end
    return A
end

function lrtest()

    D = 1000
    missing_rows = shuffle(1:(D-1))[1:10]
    P = D - length(missing_rows)

    @info "A = $D x $D , and B = $P x $P"
    @info "# rank-1 updates: $(length(missing_rows)/P)"

    # B does not contain column/row 3
    idx = setdiff(1:D, missing_rows)

    runs = 100

    t1 = zeros(runs)
    t2 = zeros(runs)
    err = zeros(runs)

    for r = 1:runs

        A = genCov(D)
        B = A[idx,idx]

        # Cholesky of A
        C = cholesky(A)

        # Cholesky of B (for testing)
        t1[r] = @elapsed CCt = cholesky(B)

        # Copy Cholesky of A
        CC = deepcopy(C)

        # rank-1 update
        t2[r] = @elapsed begin
            @inbounds for r in missing_rows
                lowrankupdate!(CC, view(CC.factors,r,(r+1):D), (r+1))
            end
        end

        # compute error
        err[r] = sum(abs.(UpperTriangular(CC.factors[idx,idx]) .- CCt.U))
    end

    @info "Nunmerical error (avg): $(mean(err))"
    @info "Time difference (avg): $(mean(t1 .- t2)) sec"
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
                        ki::Int;
                        useBLAS::Bool = true
                       ) where {T<:LinearAlgebra.BlasFloat}
    LinearAlgebra.require_one_based_indexing(A)
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
            for i = 1:k - 1
                A[k,k] -= A[i,k]'A[i,k]
            end
            Akk, info = LinearAlgebra._chol!(A[k,k], UpperTriangular)
            if info != 0
                @warn "incremental Cholesky failed"
                return info
            end
            A[k,k] = Akk
            AkkInv = inv(copy(Akk'))
            for j = k + 1:n
                for i = 1:k - 1
                    A[k,j] -= A[i,k]'A[i,j]
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
    end
    return convert(LinearAlgebra.BlasInt, 0)
end


end # module
