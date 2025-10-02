using SparseArraysBase: Unstored
using FillArrays: Zeros
using Test: @test, @testset

@testset "Unstored" begin
    a = Zeros(2, 2)
    u = Unstored(a)
    @test parent(u) ≡ a
    @test size(u) ≡ size(a)
    @test axes(u) ≡ axes(a)
end
