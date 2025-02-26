using SparseArraysBase: SparseArrayDOK
using LinearAlgebra: mul!
using Random: Random
using StableRNGs: StableRNG

const rng = StableRNG(123)

@testset "mul!" begin
  T = Float64
  szA = (2, 2)
  szB = (2, 2)
  szC = (szA[1], szB[2])

  for p in 0.0:0.25:1
    C = sprand(rng, T, szC; p)
    A = sprand(rng, T, szA; p)
    B = sprand(rng, T, szB; p)

    check1 = mul!(Array(C), Array(A), Array(B))
    @test mul!(copy(C), A, B) ≈ check1

    α = rand(rng, T)
    β = rand(rng, T) * false
    check2 = mul!(Array(C), Array(A), Array(B), α, β)
    @test mul!(copy(C), A, B, α, β) ≈ check2
  end
end
