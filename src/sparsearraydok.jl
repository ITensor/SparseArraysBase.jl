# TODO: Rewrite to use `Dictionary`.
struct SparseArrayDOK{T,N} <: AbstractSparseArray{T,N}
  storage::Dict{CartesianIndex{N},T}
  size::NTuple{N,Int}
end

function SparseArrayDOK{T,N}(size::Vararg{Int,N}) where {T,N}
  return SparseArrayDOK{T,N}(Dict{CartesianIndex{N},T}(), size)
end

function SparseArrayDOK{T}(size::Int...) where {T}
  return SparseArrayDOK{T,length(size)}(size...)
end

using Derive: @array_aliases
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
  return zero(eltype(a))
end
function setstoredindex!(a::SparseArrayDOK, value, I::Int...)
  isstored(a, I...) || throw(KeyError(CartesianIndex(I)))
  storage(a)[CartesianIndex(I)] = value
  return a
end
function setunstoredindex!(a::SparseArrayDOK, value, I::Int...)
  storage(a)[CartesianIndex(I)] = value
  return a
end

# Optional, but faster than the default.
storedpairs(a::SparseArrayDOK) = storage(a)
