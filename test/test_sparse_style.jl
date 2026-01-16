using FunctionImplementations: DefaultArrayImplementationStyle, style
using SparseArraysBase: AbstractSparseArrayImplementationStyle,
    SparseArrayImplementationStyle, sparse_style, sparsezeros
using Test: @test, @testset

module TestSparseImplementationStyleUtils
    using SparseArraysBase: AbstractSparseArray, AbstractSparseArrayImplementationStyle
    using FunctionImplementations: FunctionImplementations
    struct MySparseArrayImplementationStyle <: AbstractSparseArrayImplementationStyle end
    struct MySparseArray{T, N} <: AbstractSparseArray{T, N}
        size::NTuple{N, Int}
    end
    FunctionImplementations.ImplementationStyle(::Type{<:MySparseArray}) =
        MySparseArrayImplementationStyle()
end

@testset "Combine Sparse Styles" begin
    @test sparse_style ≡ SparseArrayImplementationStyle()
    @test ImplementationStyle(SparseArrayImplementationStyle(), SparseArrayImplementationStyle()) ≡ SparseArrayImplementationStyle()
    @test ImplementationStyle(TestSparseImplementationStyleUtils.MySparseArrayImplementationStyle(), SparseArrayImplementationStyle()) ≡
        SparseArrayImplementationStyle()
    @test ImplementationStyle(SparseArrayImplementationStyle(), TestSparseImplementationStyleUtils.MySparseArrayImplementationStyle()) ≡
        SparseArrayImplementationStyle()
    @test style(TestSparseImplementationStyleUtils.MySparseArray{Float64, 2}((2, 2))) ≡
        TestSparseImplementationStyleUtils.MySparseArrayImplementationStyle()
    @test style(sparsezeros(2, 2), TestSparseImplementationStyleUtils.MySparseArray{Float64, 2}((2, 2))) ≡
        SparseArrayImplementationStyle()
    # Regression tests for ambiguity caused by combining AbstractSparseArrayStyle with
    # DefaultArrayStyle.
    @test ImplementationStyle(
        TestSparseImplementationStyleUtils.MySparseArrayImplementationStyle(),
        DefaultArrayStyle(),
    ) ≡
        TestSparseImplementationStyleUtils.MySparseArrayImplementationStyle()
    @test style(TestSparseImplementationStyleUtils.MySparseArray{Float64, 2}((2, 2)), randn(2, 2)) ≡
        TestSparseImplementationStyleUtils.MySparseArrayImplementationStyle()
end
