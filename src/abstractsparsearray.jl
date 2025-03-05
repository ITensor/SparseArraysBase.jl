using Dictionaries: AbstractDictionary

abstract type AbstractSparseArray{T,N} <: AbstractArray{T,N} end

using DerivableInterfaces: @array_aliases
# Define AbstractSparseVector, AnyAbstractSparseArray, etc.
@array_aliases AbstractSparseArray

using DerivableInterfaces: DerivableInterfaces
function DerivableInterfaces.interface(::Type{<:AbstractSparseArray})
  return SparseArrayInterface()
end

using DerivableInterfaces: @derive

# TODO: These need to be loaded since `AbstractArrayOps`
# includes overloads of functions from these modules.
# Ideally that wouldn't be needed and can be circumvented
# with `GlobalRef`.
using ArrayLayouts: ArrayLayouts
using LinearAlgebra: LinearAlgebra

# DerivableInterfaces `Base.getindex`, `Base.setindex!`, etc.
# TODO: Define `AbstractMatrixOps` and overload for
# `AnyAbstractSparseMatrix` and `AnyAbstractSparseVector`,
# which is where matrix multiplication and factorizations
# should go.
@derive AnyAbstractSparseArray AbstractArrayOps

function Base.replace_in_print_matrix(
  A::AnyAbstractSparseArray{<:Any,2}, i::Integer, j::Integer, s::AbstractString
)
  return isstored(A, CartesianIndex(i, j)) ? s : Base.replace_with_centered_mark(s)
end

# Special-purpose constructors
# ----------------------------

"""
    sparse(storage::Union{AbstractDict,AbstractDictionary}, dims...[; getunstored])

Construct an `N`-dimensional [`SparseArrayDOK`](@ref) containing elements of type `T`. Both
`T` and `N` can either be supplied explicitly or be determined by the `storage` and the
length or number of `dims`.

This constructor does not take ownership of the supplied storage, and will result in an
independent container.
"""
sparse(::Union{AbstractDict,AbstractDictionary}, dims...; kwargs...)

const AbstractDictOrDictionary = Union{AbstractDict,AbstractDictionary}
# checked constructor from data: use `setindex!` to validate/convert input
function sparse(storage::AbstractDictOrDictionary, dims::Dims; kwargs...)
  A = SparseArrayDOK{valtype(storage)}(undef, dims; kwargs...)
  for (i, v) in pairs(storage)
    A[i] = v
  end
  return A
end
function sparse(storage::AbstractDictOrDictionary, dims::Int...; kwargs...)
  return sparse(storage, dims; kwargs...)
end

using Random: Random, AbstractRNG, default_rng

@doc """
    sparsezeros([T::Type], dims[; getunstored]) -> A::SparseArrayDOK{T}

Create an empty size `dims` sparse array.
The optional `T` argument specifies the element type, which defaults to `Float64`.
""" sparsezeros

function sparsezeros(::Type{T}, dims::Dims; kwargs...) where {T}
  return SparseArrayDOK{T}(undef, dims; kwargs...)
end
sparsezeros(::Type{T}, dims::Int...; kwargs...) where {T} = sparsezeros(T, dims; kwargs...)
sparsezeros(dims::Dims; kwargs...) = sparsezeros(Float64, dims; kwargs...)
sparsezeros(dims::Int...; kwargs...) = sparsezeros(Float64, dims; kwargs...)

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
  rng::AbstractRNG, A::AbstractArray; density::Real=0.5, randfun::Function=Random.rand
)
  ArrayLayouts.zero!(A)
  rand_inds = Random.randsubseq(rng, eachindex(A), density)
  rand_entries = randfun(rng, eltype(A), length(rand_inds))
  @inbounds for (I, v) in zip(rand_inds, rand_entries)
    A[I] = v
  end
end
