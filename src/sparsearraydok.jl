using Accessors: @set
using Dictionaries: Dictionary, IndexError, set!

function default_getunstoredindex(a::AbstractArray, I::Int...)
  return zero(eltype(a))
end

struct SparseArrayDOK{T,N,F} <: AbstractSparseArray{T,N}
  storage::Dictionary{CartesianIndex{N},T}
  size::NTuple{N,Int}
  getunstoredindex::F
end

function set_getunstoredindex(a::SparseArrayDOK, f)
  @set a.getunstoredindex = f
  return a
end

using DerivableInterfaces: DerivableInterfaces
# This defines the destination type of various operations in DerivableInterfaces.jl.

Base.similar(::AbstractSparseArrayInterface, T::Type, ax) = SparseArrayDOK{T}(undef, ax)

function SparseArrayDOK{T,N}(size::Vararg{Int,N}) where {T,N}
  getunstoredindex = default_getunstoredindex
  F = typeof(getunstoredindex)
  return SparseArrayDOK{T,N,F}(Dictionary{CartesianIndex{N},T}(), size, getunstoredindex)
end

function SparseArrayDOK{T}(::UndefInitializer, size::Tuple{Vararg{Int}}) where {T}
  return SparseArrayDOK{T,length(size)}(size...)
end

function SparseArrayDOK{T}(size::Int...) where {T}
  return SparseArrayDOK{T,length(size)}(size...)
end

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
