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

# undef constructors
function SparseArrayDOK{T}(
  ::UndefInitializer, dims::Dims, getunstoredindex=default_getunstoredindex
) where {T}
  all(≥(0), dims) || throw(ArgumentError("Invalid dimensions: $dims"))
  N = length(dims)
  F = typeof(getunstoredindex)
  return SparseArrayDOK{T,N,F}(undef, dims, getunstoredindex)
end
function SparseArrayDOK{T}(::UndefInitializer, dims::Int...) where {T}
  return SparseArrayDOK{T}(undef, dims)
end

# checked constructor from data: use `setindex!` to validate input
# does not take ownership of `storage`!
function SparseArrayDOK(
  storage::Union{AbstractDictionary{I,T},AbstractDict{I,T}}, dims::Dims{N}, unstored...
) where {N,I<:Union{Int,CartesianIndex{N}},T}
  A = SparseArrayDOK{T}(undef, dims, unstored...)
  for (i, v) in pairs(storage)
    A[i] = v
  end
  return A
end

function SparseArrayDOK{T}(::UndefInitializer, axes::Tuple) where {T}
  return SparseArrayDOK{T}(undef, Base.to_shape(axes))
end

function set_getunstoredindex(a::SparseArrayDOK, f)
  @set a.getunstoredindex = f
  return a
end

Base.similar(::AbstractSparseArrayInterface, T::Type, ax) = SparseArrayDOK{T}(undef, ax)

using DerivableInterfaces: @array_aliases
# Define `SparseMatrixDOK`, `AnySparseArrayDOK`, etc.
@array_aliases SparseArrayDOK

storage(a::SparseArrayDOK) = a.storage
Base.size(a::SparseArrayDOK) = a.size

storedvalues(a::SparseArrayDOK) = values(storage(a))
function isstored(a::SparseArrayDOK, I::Int...)
  return CartesianIndex(I) in keys(storage(a))
end
function eachstoredindex(::IndexCartesian, a::SparseArrayDOK)
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
