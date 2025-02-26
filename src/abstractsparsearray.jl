abstract type AbstractSparseArray{T,N} <: AbstractArray{T,N} end

using DerivableInterfaces: @array_aliases
# Define AbstractSparseVector, AnyAbstractSparseArray, etc.
@array_aliases AbstractSparseArray

using DerivableInterfaces: DerivableInterfaces
function DerivableInterfaces.interface(::Type{A}) where {A<:AnyAbstractSparseArray}
  return SparseArrayInterface()
end

using DerivableInterfaces: @derive
using ArrayLayouts: ArrayLayouts
using LinearAlgebra: LinearAlgebra

# This type alias is a temporary workaround since `@derive`
# doesn't parse the `@MIME_str` macro properly at the moment.
const MIMEtextplain = MIME"text/plain"

@derive (T=AnyAbstractSparseArray,) begin
  Base.getindex(::T, ::Int...)
  Base.setindex!(::T, ::Any, ::Int...)
  Base.similar(::T, ::Type, ::Tuple{Vararg{Int}})
  Base.similar(::T, ::Type, ::Tuple{Base.OneTo,Vararg{Base.OneTo}})
  # Base.copy(::T)
  # Base.copy!(::AbstractArray, ::T)
  # Base.copyto!(::AbstractArray, ::T)
  Base.map(::Any, ::T...)
  Base.map!(::Any, ::AbstractArray, ::T...)
  # Base.mapreduce(::Any, ::Any, ::T...; kwargs...)
  # Base.reduce(::Any, ::T...; kwargs...)
  # Base.all(::Function, ::T)
  # Base.all(::T)
  Base.iszero(::T)
  Base.real(::T)
  Base.fill!(::T, ::Any)
  ArrayLayouts.zero!(::T)
  Base.zero(::T)
  Base.permutedims!(::Any, ::T, ::Any)
  Broadcast.BroadcastStyle(::Type{<:T})
  Base.copyto!(::T, ::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}})
  Base.cat(::T...; kwargs...)
  ArrayLayouts.MemoryLayout(::Type{<:T})
  LinearAlgebra.mul!(::AbstractMatrix, ::T, ::T, ::Number, ::Number)
  # Base.show(::IO, ::MIMEtextplain, ::T)
end

function Base.replace_in_print_matrix(
  A::AnyAbstractSparseArray{<:Any,2}, i::Integer, j::Integer, s::AbstractString
)
  return isstored(A, CartesianIndex(i, j)) ? s : Base.replace_with_centered_mark(s)
end

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
    sprand([rng], [T::Type], dims; density::Real=0.5, rfn::Function=rand) -> A::SparseArrayDOK{T}

Create a random size `dims` sparse array in which the probability of any element being stored is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `T` argument specifies the element type, which defaults to `Float64`.
The optional `rfn` argument can be used to control the type of random elements.

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
    sprand!([rng], A::AbstractArray; density::Real=0.5, rfn::Function=rand) -> A

Overwrite part of an array with random entries, where the probability of overwriting is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `rfn` argument can be used to control the type of random elements.

See also [`sprand`](@ref).
""" sprand!

sprand!(A::AbstractArray; kwargs...) = sprand!(default_rng(), A; kwargs...)
function sprand!(
  rng::AbstractRNG, A::AbstractArray; density::Real=0.5, rfn::Function=Random.rand
)
  ArrayLayouts.zero!(A)
  rand_inds = Random.randsubseq(rng, eachindex(A), density)
  rand_entries = rfn(rng, eltype(A), length(rand_inds))
  for (I, v) in zip(rand_inds, rand_entries)
    A[I] = v
  end
end
