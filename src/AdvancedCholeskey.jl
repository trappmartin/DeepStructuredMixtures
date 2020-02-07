"""
# AdvancedCholesky module

This module aims to implement extensions to existing Cholesky factorisations
available in Julia.

"""
module AdvancedCholesky
using LinearAlgebra, Statistics, Random
import LinearAlgebra.lowrankupdate

genCov(D::Int; uplo::Symbol=:U) = Symmetric(rand(D,D) .+ Matrix(I*D,D,D), uplo)


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
            for j = i + 1:n
                @inbounds begin
                    Aji = A[j,i]
                    vj  = v[j-(k-1)]
                    A[j,i]  =   c*Aji + s*vj
                    v[j-(k-1)]    = -s'*Aji + c*vj
                end
            end
        end
    end
    return A
end

function lrtest(;D = 1000, uplo = :L)

    missing_rows = shuffle(1:(D-1))[1:10]
    P = D - length(missing_rows)

    @info "A = $D x $D , and B = $P x $P"
    @info "# rank-1 updates: $(length(missing_rows)/P)"

    # B does not contain column/row 3
    idx = setdiff(1:D, missing_rows)

    runs = 10

    t1 = zeros(runs)
    t2 = zeros(runs)
    err = zeros(runs)

    for r = 1:runs

        A = genCov(D, uplo = uplo)
        B = A[idx,idx]

        # Cholesky of A
        C = cholesky(A)

        # Cholesky of B (for testing)
        t1[r] = @elapsed CCt = cholesky(Symmetric(B, uplo))

        # Copy Cholesky of A
        CC = deepcopy(C)

        # rank-1 update
        t2[r] = @elapsed begin
            @inbounds for r in missing_rows
                if uplo == :U
                    lowrankupdate!(CC, view(CC.factors,r,(r+1):D), (r+1))
                else
                    lowrankupdate!(CC, view(CC.factors,(r+1):D,r), (r+1))
                end
            end
        end

        # compute error
        F = Cholesky(CC.factors[idx, idx], uplo, 0)
        err[r] = sum(abs.(F.U .- CCt.U))
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
function test_chol_continue()
    D = 100
    P = 10
    Σ = genCov(D, uplo = :L)
    U = deepcopy(Σ)
    A = deepcopy(Σ)

    t1 = @elapsed LinearAlgebra.cholesky!(U)
    t2 = @elapsed begin
        LAPACK.potrf!('L', view(A.data, 1:P, 1:P))
        chol_continue!(A.data, P+1, LowerTriangular)
    end

    return t1 - t2, sum(abs.(LowerTriangular(A) .- LowerTriangular(U)))
end

"""
# Continue a Cholesky decomposition of `A` from `ki` on.
`chol_continue!(A::AbstractMatrix, ki::Int)`

Currently only works if a A contains elements of a lower-triangular Cholsky decomposition.

## Usage:

```julia
A = genCov(10)
LAPACK.potrf!('L', view(A.data, 1:5, 1:5))
chol_continue!(A.data, 5+1)
```

"""
function chol_continue!(A::AbstractMatrix{T}, ki::Int) where {T<:LinearAlgebra.BlasFloat}

    # clean up
    # necessary?
    tril!(A)

    N = size(A,1)-ki
    # update lower matrix
    v = @view A[1:(ki-1),1:(ki-1)]
    @inbounds A[ki:end, 1:(ki-1)] /= v'

    # symmetrix rank-k update
    # check if this correct!
    v = @view A[ki:end,1:(ki-1)]
    C = @view A[ki:end,ki:end]
    BLAS.syrk!('L', 'N', -one(T), v, one(T), C)

    # solve Cholesky of remainder
    C = @view A[ki:end,ki:end]
    _, info = LAPACK.potrf!('L', C)

    return LowerTriangular(A), info
end

end # module
