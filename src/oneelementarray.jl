using FillArrays: Fill

# Like [`FillArrays.OneElement`](https://github.com/JuliaArrays/FillArrays.jl)
# and [`OneHotArrays.OneHotArray`](https://github.com/FluxML/OneHotArrays.jl).
struct OneElementArray{T,N,I,A,F} <: AbstractSparseArray{T,N}
  value::T
  index::I
  axes::A
  getunstoredindex::F
end

using DerivableInterfaces: @array_aliases
# Define `OneElementMatrix`, `AnyOneElementArray`, etc.
@array_aliases OneElementArray

function OneElementArray{T,N}(
  value, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}, getunstoredindex
) where {T,N}
  return OneElementArray{T,N,typeof(index),typeof(axes),typeof(getunstoredindex)}(
    value, index, axes, getunstoredindex
  )
end

function OneElementArray{T,N}(
  value, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {T,N}
  return OneElementArray{T,N}(value, index, axes, default_getunstoredindex)
end
function OneElementArray{<:Any,N}(
  value::T, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {T,N}
  return OneElementArray{T,N}(value, index, axes)
end
function OneElementArray(
  value::T, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {T,N}
  return OneElementArray{T,N}(value, index, axes)
end

function OneElementArray{T,N}(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {T,N}
  return OneElementArray{T,N}(one(T), index, axes)
end
function OneElementArray{<:Any,N}(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {N}
  return OneElementArray{Bool,N}(index, axes)
end
function OneElementArray{T}(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {T,N}
  return OneElementArray{T,N}(index, axes)
end
function OneElementArray(index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}) where {N}
  return OneElementArray{Bool,N}(index, axes)
end

function OneElementArray{T,N}(
  value, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}
) where {T,N}
  return OneElementArray{T,N}(value, last.(ax_ind), first.(ax_ind))
end
function OneElementArray{<:Any,N}(
  value::T, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}
) where {T,N}
  return OneElementArray{T,N}(value, ax_ind...)
end
function OneElementArray{T}(
  value, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}
) where {T,N}
  return OneElementArray{T,N}(value, ax_ind...)
end
function OneElementArray(
  value::T, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}
) where {T,N}
  return OneElementArray{T,N}(value, ax_ind...)
end

function OneElementArray{T,N}(ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}) where {T,N}
  return OneElementArray{T,N}(last.(ax_ind), first.(ax_ind))
end
function OneElementArray{<:Any,N}(ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}) where {N}
  return OneElementArray{Bool,N}(ax_ind...)
end
function OneElementArray{T}(ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}) where {T,N}
  return OneElementArray{T,N}(ax_ind...)
end
function OneElementArray(ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}) where {N}
  return OneElementArray{Bool,N}(ax_ind...)
end

# Fix ambiguity errors.
function OneElementArray{T,0}(value, index::Tuple{}, axes::Tuple{}) where {T}
  return OneElementArray{T,0}(value, index, axes, default_getunstoredindex)
end
function OneElementArray{<:Any,0}(value::T, index::Tuple{}, axes::Tuple{}) where {T}
  return OneElementArray{T,0}(value, index, axes)
end
function OneElementArray{T}(value, index::Tuple{}, axes::Tuple{}) where {T}
  return OneElementArray{T,0}(value, index, axes)
end
function OneElementArray(value::T, index::Tuple{}, axes::Tuple{}) where {T}
  return OneElementArray{T,0}(value, index, axes)
end

# Fix ambiguity errors.
function OneElementArray{T,0}(index::Tuple{}, axes::Tuple{}) where {T}
  return OneElementArray{T,0}(one(T), index, axes)
end
function OneElementArray{<:Any,0}(index::Tuple{}, axes::Tuple{})
  return OneElementArray{Bool,0}(index, axes)
end
function OneElementArray{T}(index::Tuple{}, axes::Tuple{}) where {T}
  return OneElementArray{T,0}(index, axes)
end
function OneElementArray(index::Tuple{}, axes::Tuple{})
  return OneElementArray{Bool,0}(value, index, axes)
end

function OneElementArray{T,0}(value) where {T}
  return OneElementArray{T,0}(value, (), ())
end
function OneElementArray{<:Any,0}(value::T) where {T}
  return OneElementArray{T,0}(value)
end
function OneElementArray{T}(value) where {T}
  return OneElementArray{T,0}(value)
end
function OneElementArray(value::T) where {T}
  return OneElementArray{T}(value)
end

function OneElementArray{T,0}() where {T}
  return OneElementArray{T,0}((), ())
end
function OneElementArray{<:Any,0}()
  return OneElementArray{Bool,0}(value)
end
function OneElementArray{T}() where {T}
  return OneElementArray{T,0}()
end
function OneElementArray()
  return OneElementArray{Bool}()
end

function OneElementArray{T,N}(
  value, index::NTuple{N,Int}, size::NTuple{N,Integer}
) where {T,N}
  return OneElementArray{T,N}(value, index, Base.oneto.(size))
end
function OneElementArray{<:Any,N}(
  value::T, index::NTuple{N,Int}, size::NTuple{N,Integer}
) where {T,N}
  return OneElementArray{T,N}(value, index, size)
end
function OneElementArray{T}(
  value, index::NTuple{N,Int}, size::NTuple{N,Integer}
) where {T,N}
  return OneElementArray{T,N}(value, index, size)
end
function OneElementArray(
  value::T, index::NTuple{N,Int}, size::NTuple{N,Integer}
) where {T,N}
  return OneElementArray{T,N}(value, index, Base.oneto.(size))
end

function OneElementArray{T,N}(index::NTuple{N,Int}, size::NTuple{N,Integer}) where {T,N}
  return OneElementArray{T,N}(one(T), index, size)
end
function OneElementArray{<:Any,N}(index::NTuple{N,Int}, size::NTuple{N,Integer}) where {N}
  return OneElementArray{Bool,N}(index, size)
end
function OneElementArray{T}(index::NTuple{N,Int}, size::NTuple{N,Integer}) where {T,N}
  return OneElementArray{T,N}(index, size)
end
function OneElementArray(index::NTuple{N,Int}, size::NTuple{N,Integer}) where {N}
  return OneElementArray{Bool,N}(index, size)
end

function OneElementVector{T}(value, index::Int, length::Integer) where {T}
  return OneElementVector{T}(value, (index,), (length,))
end
function OneElementVector(value::T, index::Int, length::Integer) where {T}
  return OneElementVector{T}(value, index, length)
end
function OneElementArray{T}(value, index::Int, length::Integer) where {T}
  return OneElementVector{T}(value, index, length)
end
function OneElementArray(value::T, index::Int, length::Integer) where {T}
  return OneElementVector{T}(value, index, length)
end

function OneElementVector{T}(index::Int, size::Integer) where {T}
  return OneElementVector{T}((index,), (size,))
end
function OneElementVector(index::Int, length::Integer)
  return OneElementVector{Bool}(index, length)
end
function OneElementArray{T}(index::Int, size::Integer) where {T}
  return OneElementVector{T}(index, size)
end
OneElementArray(index::Int, size::Integer) = OneElementVector{Bool}(index, size)

# Interface to overload for constructing arrays like `OneElementArray`,
# that may not be `OneElementArray` (i.e. wrapped versions).
function oneelement(
  value, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {N}
  return OneElementArray(value, index, axes)
end
function oneelement(
  eltype::Type, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}
) where {N}
  return oneelement(one(eltype), index, axes)
end
function oneelement(index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}) where {N}
  return oneelement(Bool, index, axes)
end

function oneelement(value, index::NTuple{N,Int}, size::NTuple{N,Integer}) where {N}
  return oneelement(value, index, Base.oneto.(size))
end
function oneelement(eltype::Type, index::NTuple{N,Int}, size::NTuple{N,Integer}) where {N}
  return oneelement(one(eltype), index, size)
end
function oneelement(index::NTuple{N,Int}, size::NTuple{N,Integer}) where {N}
  return oneelement(Bool, index, size)
end

function oneelement(value, ax_ind::Pair{<:AbstractUnitRange,Int}...)
  return oneelement(value, last.(ax_ind), first.(ax_ind))
end
function oneelement(eltype::Type, ax_ind::Pair{<:AbstractUnitRange,Int}...)
  return oneelement(one(eltype), ax_ind...)
end
function oneelement(ax_ind::Pair{<:AbstractUnitRange,Int}...)
  return oneelement(Bool, ax_ind...)
end

function oneelement(value)
  return oneelement(value, (), ())
end
function oneelement(eltype::Type)
  return oneelement(one(eltype))
end
function oneelement()
  return oneelement(Bool)
end

Base.axes(a::OneElementArray) = getfield(a, :axes)
Base.size(a::OneElementArray) = length.(axes(a))
storedvalue(a::OneElementArray) = getfield(a, :value)
storedvalues(a::OneElementArray) = Fill(storedvalue(a), 1)

storedindex(a::OneElementArray) = getfield(a, :index)
function isstored(a::OneElementArray, I::Int...)
  return I == storedindex(a)
end
function eachstoredindex(a::OneElementArray)
  return Fill(CartesianIndex(storedindex(a)), 1)
end

function getstoredindex(a::OneElementArray, I::Int...)
  return storedvalue(a)
end
function getunstoredindex(a::OneElementArray, I::Int...)
  return a.getunstoredindex(a, I...)
end
function setstoredindex!(a::OneElementArray, value, I::Int...)
  return error("`OneElementArray` is immutable, you can't set elements.")
end
function setunstoredindex!(a::OneElementArray, value, I::Int...)
  return error("`OneElementArray` is immutable, you can't set elements.")
end
