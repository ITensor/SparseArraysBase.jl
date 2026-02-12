using Aqua: Aqua
using SparseArraysBase: SparseArraysBase
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(SparseArraysBase; ambiguities = false)
end
