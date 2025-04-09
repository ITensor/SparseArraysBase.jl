using Accessors: @set
using DerivableInterfaces: DerivableInterfaces, @interface, interface, zero!
using Dictionaries: Dictionary, IndexError, set!

function getzero(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return zero(eltype(a))
end

const DOKStorage{T,N} = Dictionary{CartesianIndex{N},T}

function _SparseArrayDOK end

"""
    SparseArrayDOK{T,N,F} <: AbstractSparseArray{T,N}

`N`-dimensional sparse Dictionary-of-keys (DOK) array with elements of type `T`,
optionally with a function of type `F` to instantiate non-stored elements.
"""
struct SparseArrayDOK{T,N,F} <: AbstractSparseArray{T,N}
  storage::DOKStorage{T,N}
  size::NTuple{N,Int}
  getunstored::F
  global @inline function _SparseArrayDOK(
    storage::DOKStorage{T,N}, size::Dims{N}, getunstored::F
  ) where {T,N,F}
    return new{T,N,F}(storage, size, getunstored)
  end
end

# Constructors
# ------------
"""
    SparseArrayDOK{T}(undef, dims...[; getunstored])
    SparseArrayDOK{T,N}(undef, dims...[; getunstored])

Construct an uninitialized `N`-dimensional [`SparseArrayDOK`](@ref) containing
elements of type `T`. `N` can either be supplied explicitly, or be determined by
the length or number of `dims`.
"""
SparseArrayDOK{T,N}(::UndefInitializer, dims; kwargs...)

function SparseArrayDOK{T,N}(
  ::UndefInitializer, dims::Dims; getunstored=getzero
) where {T,N}
  (length(dims) == N && all(≥(0), dims)) ||
    throw(ArgumentError("Invalid dimensions: $dims"))
  storage = DOKStorage{T,N}()
  return _SparseArrayDOK(storage, dims, getunstored)
end
function SparseArrayDOK{T,N}(::UndefInitializer, dims::Vararg{Int,N}; kwargs...) where {T,N}
  return SparseArrayDOK{T,N}(undef, dims; kwargs...)
end
function SparseArrayDOK{T}(::UndefInitializer, dims::Dims{N}; kwargs...) where {T,N}
  return SparseArrayDOK{T,N}(undef, dims; kwargs...)
end
function SparseArrayDOK{T}(::UndefInitializer, dims::Vararg{Int,N}; kwargs...) where {T,N}
  return SparseArrayDOK{T,N}(undef, dims; kwargs...)
end

function set_getunstored(a::SparseArrayDOK, f)
  @set a.getunstored = f
  return a
end

using DerivableInterfaces: DerivableInterfaces
# This defines the destination type of various operations in DerivableInterfaces.jl.
DerivableInterfaces.arraytype(::AbstractSparseArrayInterface, T::Type) = SparseArrayDOK{T}

using DerivableInterfaces: @array_aliases
# Define `SparseMatrixDOK`, `AnySparseArrayDOK`, etc.
@array_aliases SparseArrayDOK

storage(a::SparseArrayDOK) = a.storage
Base.size(a::SparseArrayDOK) = a.size

storedvalues(a::SparseArrayDOK) = values(storage(a))
@inline function isstored(a::SparseArrayDOK{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return haskey(storage(a), CartesianIndex(I))
end
function eachstoredindex(::IndexCartesian, a::SparseArrayDOK)
  return keys(storage(a))
end
@inline function getstoredindex(a::SparseArrayDOK{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return storage(a)[CartesianIndex(I)]
end
@inline function getunstoredindex(a::SparseArrayDOK{<:Any,N}, I::Vararg{Int,N}) where {N}
  @boundscheck checkbounds(a, I...)
  return a.getunstored(a, I...)
end
@inline function setstoredindex!(
  a::SparseArrayDOK{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  # `isstored` includes a boundscheck as well
  @boundscheck isstored(a, I...) ||
    throw(IndexError(lazy"key $(CartesianIndex(I...)) not found"))
  # TODO: If `iszero(value)`, unstore the index.
  storage(a)[CartesianIndex(I)] = value
  return a
end
@inline function setunstoredindex!(
  a::SparseArrayDOK{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  @boundscheck checkbounds(a, I...)
  insert!(storage(a), CartesianIndex(I), value)
  return a
end

# Optional, but faster than the default.
storedpairs(a::SparseArrayDOK) = pairs(storage(a))

# TODO: Also handle wrappers.
function DerivableInterfaces.zero!(a::SparseArrayDOK)
  empty!(storage(a))
  return a
end
function ArrayLayouts.zero!(a::SparseArrayDOK)
  return zero!(a)
end
