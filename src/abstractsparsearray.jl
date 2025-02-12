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
using Random: Random, AbstractRNG, default_rng

@doc """
    sparsezeros([T::Type], dims) -> A::SparseArrayDOK{T}

Create an empty size `dims` sparse array.
The optional `T` argument specifies the element type, which defaults to `Float64`.
""" sparsezeros

sparsezeros(dims::Dims) = sparsezeros(Float64, dims)
sparsezeros(::Type{T}, dims::Dims) where {T} = SparseArrayDOK{T}(undef, dims)

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
sparserand(dims::Dims; kwargs...) = sparserand(default_rng(), Float64, dims; kwargs...)
function sparserand(rng::AbstractRNG, dims::Dims; kwargs...)
  return sparserand(rng, Float64, dims; kwargs...)
end
function sparserand(rng::AbstractRNG, ::Type{T}, dims::Dims; kwargs...) where {T}
  A = SparseArrayDOK{T}(undef, dims)
  sparserand!(rng, A; kwargs...)
  return A
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
