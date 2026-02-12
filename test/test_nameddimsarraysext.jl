using NamedDimsArrays: NamedDimsArray, denamed, inds, nameddims, namedoneto
using SparseArraysBase: dense, oneelement, sparsezeros
using Test: @test, @testset

@testset "SparseArraysBaseExt (eltype=$elt)" for elt in (Float64, ComplexF64)
    @testset "oneelement" begin
        i = namedoneto(3, "i")
        a = oneelement(i => 2)
        @test a isa NamedDimsArray{Bool}
        @test ndims(a) == 1
        @test issetequal(inds(a), (i,))
        @test eltype(a) ≡ Bool
        @test a[1] == 0
        @test a[2] == 1
        @test a[3] == 0

        i = namedoneto(3, "i")
        a = oneelement(elt, i => 2)
        @test a isa NamedDimsArray{elt}
        @test ndims(a) == 1
        @test issetequal(inds(a), (i,))
        @test eltype(a) ≡ elt
        @test a[1] == 0
        @test a[2] == 1
        @test a[3] == 0
    end
    @testset "dense" begin
        s = sparsezeros(elt, 3, 4)
        a = nameddims(s, (:a, :b))
        b = dense(a)
        @test denamed(b) == dense(denamed(a))
        @test denamed(b) isa Array{elt, 2}
        @test inds(b) == inds(a)
    end
end
