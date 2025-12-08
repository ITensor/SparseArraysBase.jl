using SparseArrays: SparseMatrixCSC, findnz, nnz
using SparseArraysBase: SparseMatrixDOK, eachstoredindex, isstored, sparsezeros,
    storedlength
using TensorAlgebra: contract, matricize
using Test: @testset, @test

@testset "TensorAlgebraExt (eltype = $elt)" for elt in (Float32, ComplexF64)
    a = sparsezeros(elt, (2, 2, 2))
    a[1, 1, 1] = 1
    a[2, 1, 2] = 2

    # matricize
    m = matricize(a, (1, 3), (2,))
    @test m isa SparseMatrixCSC{elt}
    @test nnz(m) == 2
    @test isstored(m, 1, 1)
    @test m[1, 1] ≡ elt(1)
    @test isstored(m, 4, 1)
    @test m[4, 1] ≡ elt(2)
    @test issetequal(eachstoredindex(m), [CartesianIndex(1, 1), CartesianIndex(4, 1)])
    for I in setdiff(CartesianIndices(m), [CartesianIndex(1, 1), CartesianIndex(4, 1)])
        @test m[I] ≡ zero(elt)
    end

    # contract
    b, l = contract(a, ("i", "j", "k"), a, ("j", "k", "l"))
    @test b isa SparseMatrixDOK{elt}
    @test storedlength(b) == 1
    @test only(eachstoredindex(b)) == CartesianIndex(1, 1)
    @test b[1, 1] ≡ elt(1)
end
