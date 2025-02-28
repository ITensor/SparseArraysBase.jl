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

# This type alias is a temporary workaround since `@derive`
# doesn't parse the `@MIME_str` macro properly at the moment.
const MIMEtextplain = MIME"text/plain"
@derive (T=AnyAbstractSparseArray,) begin
  Base.show(::IO, ::MIMEtextplain, ::T)
end

# Wraps a sparse array but replaces the unstored values.
# This is used in printing in order to customize printing
# of zero/unstored values.
struct ReplacedUnstoredSparseArray{T,N,F,Parent<:AbstractArray{T,N}} <:
       AbstractSparseArray{T,N}
  parent::Parent
  getunstoredindex::F
end
Base.parent(a::ReplacedUnstoredSparseArray) = a.parent
Base.size(a::ReplacedUnstoredSparseArray) = size(parent(a))
function isstored(a::ReplacedUnstoredSparseArray, I::Int...)
  return isstored(parent(a), I...)
end
function getstoredindex(a::ReplacedUnstoredSparseArray, I::Int...)
  return getstoredindex(parent(a), I...)
end
function getunstoredindex(a::ReplacedUnstoredSparseArray, I::Int...)
  return a.getunstoredindex(a, I...)
end
eachstoredindex(a::ReplacedUnstoredSparseArray) = eachstoredindex(parent(a))
@derive ReplacedUnstoredSparseArray AbstractArrayOps

# Special-purpose constructors
# ----------------------------
using Random: Random, AbstractRNG, default_rng

@doc """
    spzeros([T::Type], dims) -> A::SparseArrayDOK{T}

Create an empty size `dims` sparse array.
The optional `T` argument specifies the element type, which defaults to `Float64`.
""" spzeros

spzeros(dims::Dims) = spzeros(Float64, dims)
spzeros(::Type{T}, dims::Dims) where {T} = SparseArrayDOK{T}(undef, dims)

@doc """
    sprand([rng], [T::Type], dims; density::Real=0.5, randfun::Function=rand) -> A::SparseArrayDOK{T}

Create a random size `dims` sparse array in which the probability of any element being stored is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `T` argument specifies the element type, which defaults to `Float64`.
The optional `randfun` argument can be used to control the type of random elements, and should support
the signature `randfun(rng, T, N)` to generate `N` entries of type `T`.


See also [`sprand!`](@ref).
""" sprand

function sprand(::Type{T}, dims::Dims; kwargs...) where {T}
  return sprand(default_rng(), T, dims; kwargs...)
end
sprand(dims::Dims; kwargs...) = sprand(default_rng(), Float64, dims; kwargs...)
function sprand(rng::AbstractRNG, dims::Dims; kwargs...)
  return sprand(rng, Float64, dims; kwargs...)
end
function sprand(rng::AbstractRNG, ::Type{T}, dims::Dims; kwargs...) where {T}
  A = SparseArrayDOK{T}(undef, dims)
  sprand!(rng, A; kwargs...)
  return A
end

@doc """
    sprand!([rng], A::AbstractArray; density::Real=0.5, randfun::Function=rand) -> A

Overwrite part of an array with random entries, where the probability of overwriting is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `randfun` argument can be used to control the type of random elements, and should support
the signature `randfun(rng, T, N)` to generate `N` entries of type `T`.

See also [`sprand`](@ref).
""" sprand!

sprand!(A::AbstractArray; kwargs...) = sprand!(default_rng(), A; kwargs...)
function sprand!(
  rng::AbstractRNG, A::AbstractArray; density::Real=0.5, randfun::Function=Random.rand
)
  ArrayLayouts.zero!(A)
  rand_inds = Random.randsubseq(rng, eachindex(A), density)
  rand_entries = randfun(rng, eltype(A), length(rand_inds))
  @inbounds for (I, v) in zip(rand_inds, rand_entries)
    A[I] = v
  end
end
