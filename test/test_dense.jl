using Adapt: adapt
using JLArrays: JLArray
using SparseArraysBase: dense, sparsezeros
using Test: @test, @testset

module TestDenseUtils
    struct MySparseArrayDOK{T, N, S <: AbstractVector{T}} <: AbstractArray{T, N}
        storedvalues::S
        storedindices::Dict{CartesianIndex{N}, Int}
        size::NTuple{N, Int}
    end
    Base.size(a::MySparseArrayDOK) = a.size
    function Base.getindex(a::MySparseArrayDOK{<:Any, N}, I::Vararg{Int, N}) where {N}
        storageindex = get(a.storedindices, CartesianIndex(I), nothing)
        isnothing(storageindex) && return zero(eltype(a))
        return a.storedvalues[storageindex]
    end
    Base.parent(a::MySparseArrayDOK) = a.storedvalues
end

elts = (Float32, ComplexF64)
arrayts = (Array, JLArray)
@testset "dense (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts, elt in elts
    dev(x) = adapt(arrayt, x)

    @testset "SparseArrayDOK" begin
        s = sparsezeros(elt, 3, 4)
        s[1, 2] = 2
        d = dense(s)
        @test d isa Matrix{elt}
        @test d == [0 2 0 0; 0 0 0 0; 0 0 0 0]
    end

    @testset "Custom sparse array" begin
        s = TestDenseUtils.MySparseArrayDOK(
            dev(elt[2, 4]), Dict([CartesianIndex(1, 2) => 1, CartesianIndex(3, 4) => 2]), (3, 4)
        )
        d = dense(s)
        @test d isa arrayt{elt, 2}
        @test d == dev(elt[0 2 0 0; 0 0 0 0; 0 0 0 4])
    end
end
