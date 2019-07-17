module AdvancedCholeskey

using LinearAlgebra
using BenchmarkTools

# A = Float64.([25 15 5; 15 18 0; -5 0 11])
# a = A[1:2,1:2]

function test(A, a)
    _A = deepcopy(A)
    _a = deepcopy(a)

    t1 = @elapsed begin
        LinearAlgebra.cholesky(Symmetric(_a, :U))
        LinearAlgebra.cholesky(Symmetric(_A, :U))
    end

    t2 = @elapsed res = begin
        U = LinearAlgebra.cholesky(Symmetric(a, :U))

        @inbounds begin
            for i in 1:size(a,2)
                for j in 1:i
                    A[j,i] = U.U[j,i]
                    A[i,j] = U.U[j,i]
                end
            end
        end
        chol2 = chol_continue!(A, UpperTriangular, 3)
    end

    @assert t1 > t2

    return t1 - t2, res
end

function chol_continue!(A::AbstractMatrix, ::Type{UpperTriangular}, ki::Int)
    @assert !LinearAlgebra.has_offset_axes(A)
    n = LinearAlgebra.checksquare(A)
    @assert ki <= n

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
