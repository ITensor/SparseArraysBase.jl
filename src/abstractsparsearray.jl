using Dictionaries: AbstractDictionary

abstract type AbstractSparseArray{T, N} <: AbstractArray{T, N} end

Base.convert(T::Type{<:AbstractSparseArray}, a::AbstractArray) = a isa T ? a : T(a)

## using DerivableInterfaces: @array_aliases
## # Define AbstractSparseVector, AnyAbstractSparseArray, etc.
## @array_aliases AbstractSparseArray

using FunctionImplementations: FunctionImplementations
function FunctionImplementations.Style(::Type{<:AbstractSparseArray})
    return SparseArrayInterface()
end

function Base.copy(a::AnyAbstractSparseArray)
    return copyto!(similar(a), a)
end

function Base.similar(a::AnyAbstractSparseArray, unstored::Unstored)
    return SparseArrayDOK(undef, unstored)
end
function Base.similar(a::AnyAbstractSparseArray)
    return similar(a, Unstored(unstored(a)))
end
function Base.similar(a::AnyAbstractSparseArray, T::Type)
    return similar(a, Unstored(unstoredsimilar(unstored(a), T)))
end
function Base.similar(a::AnyAbstractSparseArray, ax::Tuple)
    return similar(a, Unstored(unstoredsimilar(unstored(a), ax)))
end

function similar_sparsearray(a::AbstractArray, T::Type, ax::Tuple)
    return similar(a, Unstored(unstoredsimilar(unstored(a), T, ax)))
end
function Base.similar(a::AnyAbstractSparseArray, T::Type, ax::Tuple{Vararg{Int}})
    return similar_sparsearray(a, T, ax)
end
function Base.similar(
        a::AnyAbstractSparseArray, T::Type, ax::Tuple{Integer, Vararg{Integer}}
    )
    return similar_sparsearray(a, T, ax)
end
function Base.similar(
        a::AnyAbstractSparseArray,
        T::Type,
        ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}},
    )
    return similar_sparsearray(a, T, ax)
end

## using DerivableInterfaces: @derive

# TODO: These need to be loaded since `AbstractArrayOps`
# includes overloads of functions from these modules.
# Ideally that wouldn't be needed and can be circumvented
# with `GlobalRef`.
using ArrayLayouts: ArrayLayouts
using LinearAlgebra: LinearAlgebra

## @derive (T = AnyAbstractSparseArray,) begin
##     Base.getindex(::T, ::Any...)
##     Base.getindex(::T, ::Int...)
##     Base.setindex!(::T, ::Any, ::Any...)
##     Base.setindex!(::T, ::Any, ::Int...)
##     Base.copy!(::AbstractArray, ::T)
##     Base.copyto!(::AbstractArray, ::T)
##     Base.map(::Any, ::T...)
##     Base.map!(::Any, ::AbstractArray, ::T...)
##     Base.mapreduce(::Any, ::Any, ::T...; kwargs...)
##     Base.reduce(::Any, ::T...; kwargs...)
##     Base.all(::Function, ::T)
##     Base.all(::T)
##     Base.iszero(::T)
##     Base.real(::T)
##     Base.fill!(::T, ::Any)
##     DerivableInterfaces.zero!(::T)
##     Base.zero(::T)
##     Base.permutedims!(::Any, ::T, ::Any)
##     Broadcast.BroadcastStyle(::Type{<:T})
##     Base.copyto!(::T, ::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}})
##     ArrayLayouts.MemoryLayout(::Type{<:T})
##     LinearAlgebra.mul!(::AbstractMatrix, ::T, ::T, ::Number, ::Number)
## end

using FunctionImplementations.Concatenate: concatenate
# We overload `Base._cat` instead of `Base.cat` since it
# is friendlier for invalidations/compile times, see
# https://github.com/ITensor/SparseArraysBase.jl/issues/25.
function Base._cat(dims, a::AnyAbstractSparseArray...)
    return concatenate(dims, a...)
end

# TODO: Use `map(WeakPreserving(f), a)` instead.
# Currently that has trouble with type unstable maps, since
# the element type becomes abstract and therefore the zero/unstored
# values are not well defined.
function map_stored(f, a::AnyAbstractSparseArray)
    iszero(storedlength(a)) && return a
    kvs = storedpairs(a)
    # `collect` to convert to `Vector`, since otherwise
    # if it stays as `Dictionary` we might hit issues like
    # https://github.com/andyferris/Dictionaries.jl/issues/163.
    ks = collect(first.(kvs))
    vs = collect(last.(kvs))
    vs′ = map(f, vs)
    a′ = zero!(similar(a, eltype(vs′)))
    for (k, v′) in zip(ks, vs′)
        a′[k] = v′
    end
    return a′
end

using Adapt: adapt
function Base.print_array(io::IO, a::AnyAbstractSparseArray)
    # TODO: Use `map(WeakPreserving(adapt(Array)), a)` instead.
    # Currently that has trouble with type unstable maps, since
    # the element type becomes abstract and therefore the zero/unstored
    # values are not well defined.
    a′ = map_stored(adapt(Array), a)
    return @invoke Base.print_array(io::typeof(io), a′::AbstractArray{<:Any, ndims(a)})
end
function Base.replace_in_print_matrix(
        a::AnyAbstractSparseVecOrMat, i::Integer, j::Integer, s::AbstractString
    )
    return isstored(a, i, j) ? s : Base.replace_with_centered_mark(s)
end

# Special-purpose constructors
# ----------------------------

"""
    sparse(storage::Union{AbstractDict,AbstractDictionary}, dims...[; getunstored])

Construct an `N`-dimensional [`SparseArrayDOK`](@ref) containing elements of type `T`. Both
`T` and `N` can either be supplied explicitly or be determined by the `storage` and the
length or number of `dims`. If `dims` aren't specified, the size will be determined automatically
from the input indices.

This constructor does not take ownership of the supplied storage, and will result in an
independent container.
"""
sparse(::Union{AbstractDict, AbstractDictionary}, dims...)

const AbstractDictOrDictionary = Union{AbstractDict, AbstractDictionary}
# checked constructor from data: use `setindex!` to validate/convert input
function sparse(storage::AbstractDictOrDictionary, unstored::AbstractArray)
    A = SparseArrayDOK(undef, Unstored(unstored))
    for (i, v) in pairs(storage)
        A[i] = v
    end
    return A
end
function sparse(storage::AbstractDictOrDictionary, ax::Tuple)
    return sparse(storage, Zeros{valtype(storage)}(ax))
end
function sparse(storage::AbstractDictOrDictionary, dims::Int...)
    return sparse(storage, dims)
end
# Determine the size automatically.
function sparse(storage::AbstractDictOrDictionary)
    dims = ntuple(Returns(0), length(keytype(storage)))
    for I in keys(storage)
        dims = map(max, dims, Tuple(I))
    end
    return sparse(storage, dims)
end

using Random: Random, AbstractRNG, default_rng

@doc """
    sparsezeros([T::Type], dims[; getunstored]) -> A::SparseArrayDOK{T}

Create an empty size `dims` sparse array.
The optional `T` argument specifies the element type, which defaults to `Float64`.
""" sparsezeros

function sparsezeros(::Type{T}, unstored::AbstractArray{<:Any, N}) where {T, N}
    return SparseArrayDOK{T, N}(undef, Unstored(unstored))
end
function sparsezeros(unstored::AbstractArray{T, N}) where {T, N}
    return SparseArrayDOK{T, N}(undef, Unstored(unstored))
end
function sparsezeros(::Type{T}, dims::Dims) where {T}
    return sparsezeros(T, Zeros{T}(dims))
end
sparsezeros(::Type{T}, dims::Int...) where {T} = sparsezeros(T, dims)
sparsezeros(dims::Dims) = sparsezeros(Float64, dims)
sparsezeros(dims::Int...) = sparsezeros(Float64, dims)

@doc """
    sparserand([rng], [T::Type], dims; density::Real=0.5, randfun::Function=rand) -> A::SparseArrayDOK{T}

Create a random size `dims` sparse array in which the probability of any element being stored is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `T` argument specifies the element type, which defaults to `Float64`.
The optional `randfun` argument can be used to control the type of random elements, and should support
the signature `randfun(rng, T, N)` to generate `N` entries of type `T`.


See also [`sparserand!`](@ref).
""" sparserand

function sparserand(::Type{T}, dims::Dims; kwargs...) where {T}
    return sparserand(default_rng(), T, dims; kwargs...)
end
function sparserand(::Type{T}, dims::Int...; kwargs...) where {T}
    return sparserand(T, dims; kwargs...)
end
sparserand(dims::Dims; kwargs...) = sparserand(default_rng(), Float64, dims; kwargs...)
sparserand(dims::Int...; kwargs...) = sparserand(dims; kwargs...)
function sparserand(rng::AbstractRNG, dims::Dims; kwargs...)
    return sparserand(rng, Float64, dims; kwargs...)
end
function sparserand(rng::AbstractRNG, dims::Int...; kwargs...)
    return sparserand(rng, dims; kwargs...)
end
function sparserand(rng::AbstractRNG, ::Type{T}, dims::Dims; kwargs...) where {T}
    A = SparseArrayDOK{T}(undef, dims)
    sparserand!(rng, A; kwargs...)
    return A
end
function sparserand(rng::AbstractRNG, ::Type{T}, dims::Int...; kwargs...) where {T}
    return sparserand(rng, T, dims; kwargs...)
end

@doc """
    sparserand!([rng], A::AbstractArray; density::Real=0.5, randfun::Function=rand) -> A

Overwrite part of an array with random entries, where the probability of overwriting is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `randfun` argument can be used to control the type of random elements, and should support
the signature `randfun(rng, T, N)` to generate `N` entries of type `T`.

See also [`sparserand`](@ref).
""" sparserand!

sparserand!(A::AbstractArray; kwargs...) = sparserand!(default_rng(), A; kwargs...)
function sparserand!(
        rng::AbstractRNG, A::AbstractArray; density::Real = 0.5, randfun::Function = Random.rand
    )
    ArrayLayouts.zero!(A)
    rand_inds = Random.randsubseq(rng, eachindex(A), density)
    rand_entries = randfun(rng, eltype(A), length(rand_inds))
    return @inbounds for (I, v) in zip(rand_inds, rand_entries)
        A[I] = v
    end
end

# Catch some cases that aren't getting caught by the current
# DerivableInterfaces.jl logic.
# TODO: Make this more systematic once DerivableInterfaces.jl
# is rewritten.
using ArrayLayouts: ArrayLayouts, MemoryLayout
using LinearAlgebra: LinearAlgebra, Adjoint
function ArrayLayouts.MemoryLayout(::Type{Transpose{T, P}}) where {T, P <: AbstractSparseMatrix}
    return MemoryLayout(P)
end
function ArrayLayouts.MemoryLayout(::Type{Adjoint{T, P}}) where {T, P <: AbstractSparseMatrix}
    return MemoryLayout(P)
end
function LinearAlgebra.mul!(
        dest::AbstractMatrix,
        A::Adjoint{<:Any, <:AbstractSparseMatrix},
        B::AbstractSparseMatrix,
        α::Number,
        β::Number,
    )
    return ArrayLayouts.mul!(dest, A, B, α, β)
end
function LinearAlgebra.mul!(
        dest::AbstractMatrix,
        A::AbstractSparseMatrix,
        B::Adjoint{<:Any, <:AbstractSparseMatrix},
        α::Number,
        β::Number,
    )
    return ArrayLayouts.mul!(dest, A, B, α, β)
end
function LinearAlgebra.mul!(
        dest::AbstractMatrix,
        A::Adjoint{<:Any, <:AbstractSparseMatrix},
        B::Adjoint{<:Any, <:AbstractSparseMatrix},
        α::Number,
        β::Number,
    )
    return ArrayLayouts.mul!(dest, A, B, α, β)
end
