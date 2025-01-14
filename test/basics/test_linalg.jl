using SparseArraysBase: SparseArrayDOK
using LinearAlgebra: mul!
using Random

# TODO: add this to main package
function sprand(::Type{T}, sz::Base.Dims; p::Real=0.5) where {T}
  A = SparseArrayDOK{T}(undef, sz)
  for I in eachindex(A)
    if rand() < p
      A[I] = rand(T)
    end
  end
  return A
end

@testset "mul!" begin
  T = Float64
  szA = (2, 2)
  szB = (2, 2)
  szC = (szA[1], szB[2])

  for p in 0.0:0.25:1
    C = sprand(T, szC; p)
    A = sprand(T, szA; p)
    B = sprand(T, szB; p)

    check1 = mul!(Array(C), Array(A), Array(B))
    @test mul!(copy(C), A, B) ≈ check1

    α = rand()
    β = rand()
    check2 = mul!(Array(C), Array(A), Array(B), α, β)
    @test mul!(copy(C), A, B, α, β) ≈ check2
  end
end
