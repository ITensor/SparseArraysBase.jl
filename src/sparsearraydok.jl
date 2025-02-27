using Accessors: @set
using Dictionaries: AbstractDictionary, Dictionary, IndexError, set!

function default_getunstoredindex(a::AbstractArray, I::Int...)
  return zero(eltype(a))
end

const DOKStorage{T,N} = Dictionary{CartesianIndex{N},T}

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

## constructors with T and N
# -> make SparseMatrix{T}(undef, ...) work
function SparseArrayDOK{T,N}(
  ::UndefInitializer, dims::Dims, getunstoredindex=default_getunstoredindex
) where {T,N}
  (length(dims) == N && all(≥(0), dims)) ||
    throw(ArgumentError("Invalid dimensions: $dims"))
  F = typeof(getunstoredindex)
  return SparseArrayDOK{T,N,F}(undef, dims, getunstoredindex)
end

## constructors with T
function SparseArrayDOK{T}(::UndefInitializer, dims::Dims{N}, unstored...) where {T,N}
  return SparseArrayDOK{T,N}(undef, dims, unstored...)
end

function SparseArrayDOK{T}(::UndefInitializer, dims::Vararg{Int,N}) where {T,N}
  return SparseArrayDOK{T,N}(undef, dims)
end

# checked constructor from data: use `setindex!` to validate/convert input
function SparseArrayDOK{T}(
  storage::Union{AbstractDictionary,AbstractDict}, dims::Dims, unstored...
) where {T}
  A = SparseArrayDOK{T}(undef, dims, unstored...)
  for (i, v) in pairs(storage)
    A[i] = v
  end
  return A
end

## constructors without type parameters
function SparseArrayDOK(
  storage::Union{AbstractDictionary,AbstractDict}, dims::Dims, unstored...
)
  T = valtype(storage)
  return SparseArrayDOK{T}(storage, dims, unstored...)
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
