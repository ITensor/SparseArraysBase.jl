using SparseArraysBase: SparseArraysBase
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
  Aqua.test_all(SparseArraysBase)
end
