using Accessors: @set
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
function isstored(a::SparseArrayDOK, I::Int...)
  return CartesianIndex(I) in keys(storage(a))
end
function eachstoredindex(a::SparseArrayDOK)
  return keys(storage(a))
end
function getstoredindex(a::SparseArrayDOK, I::Int...)
  return storage(a)[CartesianIndex(I)]
end
function getunstoredindex(a::SparseArrayDOK, I::Int...)
  return a.getunstored(a, I...)
end
function setstoredindex!(a::SparseArrayDOK, value, I::Int...)
  # TODO: Have a way to disable this check, analogous to `checkbounds`,
  # since this is already checked in `setindex!`.
  isstored(a, I...) || throw(IndexError("key $(CartesianIndex(I)) not found"))
  # TODO: If `iszero(value)`, unstore the index.
  storage(a)[CartesianIndex(I)] = value
  return a
end
function setunstoredindex!(a::SparseArrayDOK, value, I::Int...)
  set!(storage(a), CartesianIndex(I), value)
  return a
end

# Optional, but faster than the default.
storedpairs(a::SparseArrayDOK) = pairs(storage(a))

function ArrayLayouts.zero!(a::SparseArrayDOK)
  empty!(storage(a))
  return a
end
