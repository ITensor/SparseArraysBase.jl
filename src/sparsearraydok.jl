using Accessors: @set
using Dictionaries: AbstractDictionary, Dictionary, IndexError, set!

function default_getunstoredindex(a::AbstractArray, I::Int...)
  return zero(eltype(a))
end

const DOKStorage{T,N} = Dictionary{CartesianIndex{N},T}

"""
    SparseArrayDOK{T,N,F} <: AbstractSparseArray{T,N}

`N`-dimensional sparse Dictionary-of-keys (DOK) array with elements of type `T`,
optionally with a function of type `F` to instantiate non-stored elements.
"""
struct SparseArrayDOK{T,N,F} <: AbstractSparseArray{T,N}
  storage::DOKStorage{T,N}
  size::NTuple{N,Int}
  getunstoredindex::F

  # bare constructor
  function SparseArrayDOK{T,N,F}(
    ::UndefInitializer, size::Dims{N}, getunstoredindex::F
  ) where {T,N,F}
    storage = DOKStorage{T,N}()
    return new{T,N,F}(storage, size, getunstoredindex)
  end

  # unchecked constructor from data
  function SparseArrayDOK{T,N,F}(
    storage::DOKStorage{T,N}, size::Dims{N}, getunstoredindex::F
  ) where {T,N,F}
    return new{T,N,F}(storage, size, getunstoredindex)
  end
end

# Constructors
# ------------
"""
    SparseArrayDOK{T}(undef, dims, unstored...)
    SparseArrayDOK{T}(undef, dims...)
    SparseArrayDOK{T,N}(undef, dims, unstored...)
    SparseArrayDOK{T,N}(undef, dims...)

Construct an uninitialized `N`-dimensional [`SparseArrayDOK`](@ref) containing
elements of type `T`. `N` can either be supplied explicitly, or be determined by
the length or number of `dims`.
"""
SparseArrayDOK{T,N}(::UndefInitializer, dims, unstored...)

function SparseArrayDOK{T,N}(
  ::UndefInitializer, dims::Dims, getunstoredindex=default_getunstoredindex
) where {T,N}
  (length(dims) == N && all(≥(0), dims)) ||
    throw(ArgumentError("Invalid dimensions: $dims"))
  F = typeof(getunstoredindex)
  return SparseArrayDOK{T,N,F}(undef, dims, getunstoredindex)
end
function SparseArrayDOK{T}(::UndefInitializer, dims::Dims{N}, unstored...) where {T,N}
  return SparseArrayDOK{T,N}(undef, dims, unstored...)
end
function SparseArrayDOK{T}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return SparseArrayDOK{T,N}(undef, dims)
end

"""
    SparseArrayDOK(storage::Union{AbstractDict,AbstractDictionary}, dims, unstored...)
    SparseArrayDOK{T}(storage::Union{AbstractDict,AbstractDictionary}, dims, unstored...)
    SparseArrayDOK{T,N}(storage::Union{AbstractDict,AbstractDictionary}, dims, unstored...)

Construct an `N`-dimensional [`SparseArrayDOK`](@ref) containing elements of type `T`. Both
`T` and `N` can either be supplied explicitly or be determined by the `storage` and the
length or number of `dims`.

This constructor does not take ownership of the supplied storage, and will result in an
independent container.
"""
SparseArrayDOK{T,N}(::Union{AbstractDict,AbstractDictionary}, dims, unstored...)

const AbstractDictOrDictionary = Union{AbstractDict,AbstractDictionary}
# checked constructor from data: use `setindex!` to validate/convert input
function SparseArrayDOK{T,N}(
  storage::AbstractDictOrDictionary, dims::Dims, unstored...
) where {T,N}
  A = SparseArrayDOK{T,N}(undef, dims, unstored...)
  for (i, v) in pairs(storage)
    A[i] = v
  end
  return A
end
function SparseArrayDOK{T}(
  storage::AbstractDictOrDictionary, dims::Dims, unstored...
) where {T}
  return SparseArrayDOK{T,length(dims)}(storage, dims, unstored...)
end
function SparseArrayDOK(storage::AbstractDictOrDictionary, dims::Dims, unstored...)
  return SparseArrayDOK{valtype(storage)}(storage, dims, unstored...)
end

function set_getunstoredindex(a::SparseArrayDOK, f)
  @set a.getunstoredindex = f
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
  return a.getunstoredindex(a, I...)
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
