using LinearAlgebra: Diagonal, diagind
using SparseArraysBase:
  eachstoredindex,
  getstoredindex,
  getunstoredindex,
  setstoredindex!,
  isstored,
  storedlength,
  storedpairs,
  storedvalues

using Test: @test, @testset

elts = (Float32, Float64, Complex{Float32}, Complex{Float64})

@testset "Diagonal{$T}" for T in elts
  L = 4
  D = Diagonal(rand(T, 4))
  @test storedlength(D) == 4
  @test eachstoredindex(D) == diagind(D, IndexCartesian())
  @test isstored(D, 2, 2)
  @test getstoredindex(D, 2, 2) == D[2, 2]
  @test !isstored(D, 2, 1)
  @test getunstoredindex(D, 2, 2) == zero(T)
end
