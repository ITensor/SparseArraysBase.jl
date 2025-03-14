using SparseArraysBase: sparserand
using LinearAlgebra: mul!
using StableRNGs: StableRNG
using Test: @test, @testset

const rng = StableRNG(123)

@testset "mul!" begin
  T = Float64
  szA = (2, 2)
  szB = (2, 2)
  szC = (szA[1], szB[2])

  for density in 0.0:0.25:1
    C = sparserand(rng, T, szC; density)
    A = sparserand(rng, T, szA; density)
    B = sparserand(rng, T, szB; density)

    check1 = mul!(Array(C), Array(A), Array(B))
    @test mul!(copy(C), A, B) ≈ check1

    α = rand(rng, T)
    β = rand(rng, T)
    check2 = mul!(Array(C), Array(A), Array(B), α, β)
    @test mul!(copy(C), A, B, α, β) ≈ check2
  end
end
