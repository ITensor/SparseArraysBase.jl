using FunctionImplementations: Style, style
using SparseArraysBase: AbstractSparseArrayStyle, SparseArrayStyle, sparse_style, sparsezeros
using Test: @test, @testset

module TestSparseStyleUtils
    using SparseArraysBase: AbstractSparseArray, AbstractSparseArrayStyle
    using FunctionImplementations: FunctionImplementations
    struct MySparseArrayStyle <: AbstractSparseArrayStyle end
    struct MySparseArray{T, N} <: AbstractSparseArray{T, N}
        size::NTuple{N, Int}
    end
    FunctionImplementations.Style(::Type{<:MySparseArray}) = MySparseArrayStyle()
end

@testset "Combine Sparse Styles" begin
    @test sparse_style ≡ SparseArrayStyle()
    @test Style(SparseArrayStyle(), SparseArrayStyle()) ≡ SparseArrayStyle()
    @test Style(TestSparseStyleUtils.MySparseArrayStyle(), SparseArrayStyle()) ≡
        SparseArrayStyle()
    @test Style(SparseArrayStyle(), TestSparseStyleUtils.MySparseArrayStyle()) ≡
        SparseArrayStyle()
    @test style(TestSparseStyleUtils.MySparseArray{Float64, 2}((2, 2))) ≡
        TestSparseStyleUtils.MySparseArrayStyle()
    @test style(sparsezeros(2, 2), TestSparseStyleUtils.MySparseArray{Float64, 2}((2, 2))) ≡
        SparseArrayStyle()
end
