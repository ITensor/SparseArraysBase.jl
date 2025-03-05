using FillArrays: Fill

function _OneElementArray end

# Like [`FillArrays.OneElement`](https://github.com/JuliaArrays/FillArrays.jl)
# and [`OneHotArrays.OneHotArray`](https://github.com/FluxML/OneHotArrays.jl).
struct OneElementArray{T,N,I,A,F} <: AbstractSparseArray{T,N}
  value::T
  index::I
  axes::A
  getunstoredfun::F
  global @inline function _OneElementArray(
    value::T, index::I, axes::A, getunstoredfun::F
  ) where {T,I,A,F}
    N = length(axes)
    @assert N == length(index)
    return new{T,N,I,A,F}(value, index, axes, getunstoredfun)
  end
end

using DerivableInterfaces: @array_aliases
# Define `OneElementMatrix`, `AnyOneElementArray`, etc.
@array_aliases OneElementArray

function OneElementArray{T,N}(
  value, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; getunstoredfun=getzero
) where {T,N}
  return _OneElementArray(convert(T, value), index, axes, getunstoredfun)
end

function OneElementArray{<:Any,N}(
  value::T, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, index, axes; kwargs...)
end
function OneElementArray(
  value::T, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, index, axes; kwargs...)
end

function OneElementArray{T,N}(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(one(T), index, axes; kwargs...)
end
function OneElementArray{<:Any,N}(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {N}
  return OneElementArray{Bool,N}(index, axes; kwargs...)
end
function OneElementArray{T}(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(index, axes; kwargs...)
end
function OneElementArray(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {N}
  return OneElementArray{Bool,N}(index, axes; kwargs...)
end

function OneElementArray{T,N}(
  value, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, last.(ax_ind), first.(ax_ind); kwargs...)
end
function OneElementArray{<:Any,N}(
  value::T, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, ax_ind...; kwargs...)
end
function OneElementArray{T}(
  value, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, ax_ind...; kwargs...)
end
function OneElementArray(
  value::T, ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, ax_ind...; kwargs...)
end

function OneElementArray{T,N}(
  ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(last.(ax_ind), first.(ax_ind); kwargs...)
end
function OneElementArray{<:Any,N}(
  ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {N}
  return OneElementArray{Bool,N}(ax_ind...; kwargs...)
end
function OneElementArray{T}(
  ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(ax_ind...)
end
function OneElementArray(
  ax_ind::Vararg{Pair{<:AbstractUnitRange,Int},N}; kwargs...
) where {N}
  return OneElementArray{Bool,N}(ax_ind...; kwargs...)
end

# Fix ambiguity errors.
function OneElementArray{T,0}(
  value, index::Tuple{}, axes::Tuple{}; getunstoredfun=getzero
) where {T}
  return _OneElementArray(convert(T, value), index, axes, getunstoredfun)
end
function OneElementArray{<:Any,0}(
  value::T, index::Tuple{}, axes::Tuple{}; kwargs...
) where {T}
  return OneElementArray{T,0}(value, index, axes; kwargs...)
end
function OneElementArray{T}(value, index::Tuple{}, axes::Tuple{}; kwargs...) where {T}
  return OneElementArray{T,0}(value, index, axes; kwargs...)
end
function OneElementArray(value::T, index::Tuple{}, axes::Tuple{}; kwargs...) where {T}
  return OneElementArray{T,0}(value, index, axes; kwargs...)
end

# Fix ambiguity errors.
function OneElementArray{T,0}(index::Tuple{}, axes::Tuple{}; kwargs...) where {T}
  return OneElementArray{T,0}(one(T), index, axes; kwargs...)
end
function OneElementArray{<:Any,0}(index::Tuple{}, axes::Tuple{}; kwargs...)
  return OneElementArray{Bool,0}(index, axes; kwargs...)
end
function OneElementArray{T}(index::Tuple{}, axes::Tuple{}; kwargs...) where {T}
  return OneElementArray{T,0}(index, axes; kwargs...)
end
function OneElementArray(index::Tuple{}, axes::Tuple{}; kwargs...)
  return OneElementArray{Bool,0}(value, index, axes; kwargs...)
end

function OneElementArray{T,0}(value; kwargs...) where {T}
  return OneElementArray{T,0}(value, (), (); kwargs...)
end
function OneElementArray{<:Any,0}(value::T; kwargs...) where {T}
  return OneElementArray{T,0}(value; kwargs...)
end
function OneElementArray{T}(value; kwargs...) where {T}
  return OneElementArray{T,0}(value; kwargs...)
end
function OneElementArray(value::T; kwargs...) where {T}
  return OneElementArray{T}(value; kwargs...)
end

function OneElementArray{T,0}(; kwargs...) where {T}
  return OneElementArray{T,0}((), (); kwargs...)
end
function OneElementArray{<:Any,0}(; kwargs...)
  return OneElementArray{Bool,0}(value; kwargs...)
end
function OneElementArray{T}(; kwargs...) where {T}
  return OneElementArray{T,0}(; kwargs...)
end
function OneElementArray(; kwargs...)
  return OneElementArray{Bool}(; kwargs...)
end

function OneElementArray{T,N}(
  value, index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, index, Base.oneto.(size); kwargs...)
end
function OneElementArray{<:Any,N}(
  value::T, index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, index, size; kwargs...)
end
function OneElementArray{T}(
  value, index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, index, size; kwargs...)
end
function OneElementArray(
  value::T, index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(value, index, Base.oneto.(size); kwargs...)
end

function OneElementArray{T,N}(
  index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(one(T), index, size; kwargs...)
end
function OneElementArray{<:Any,N}(
  index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {N}
  return OneElementArray{Bool,N}(index, size; kwargs...)
end
function OneElementArray{T}(
  index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {T,N}
  return OneElementArray{T,N}(index, size; kwargs...)
end
function OneElementArray(index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...) where {N}
  return OneElementArray{Bool,N}(index, size; kwargs...)
end

function OneElementVector{T}(value, index::Int, length::Integer; kwargs...) where {T}
  return OneElementVector{T}(value, (index,), (length,); kwargs...)
end
function OneElementVector(value::T, index::Int, length::Integer; kwargs...) where {T}
  return OneElementVector{T}(value, index, length; kwargs...)
end
function OneElementArray{T}(value, index::Int, length::Integer; kwargs...) where {T}
  return OneElementVector{T}(value, index, length; kwargs...)
end
function OneElementArray(value::T, index::Int, length::Integer; kwargs...) where {T}
  return OneElementVector{T}(value, index, length; kwargs...)
end

function OneElementVector{T}(index::Int, size::Integer; kwargs...) where {T}
  return OneElementVector{T}((index,), (size,); kwargs...)
end
function OneElementVector(index::Int, length::Integer; kwargs...)
  return OneElementVector{Bool}(index, length; kwargs...)
end
function OneElementArray{T}(index::Int, size::Integer; kwargs...) where {T}
  return OneElementVector{T}(index, size; kwargs...)
end
function OneElementArray(index::Int, size::Integer; kwargs...)
  return OneElementVector{Bool}(index, size; kwargs...)
end

# Interface to overload for constructing arrays like `OneElementArray`,
# that may not be `OneElementArray` (i.e. wrapped versions).
function oneelement(
  value, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {N}
  return OneElementArray(value, index, axes; kwargs...)
end
function oneelement(
  eltype::Type, index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {N}
  return oneelement(one(eltype), index, axes; kwargs...)
end
function oneelement(
  index::NTuple{N,Int}, axes::NTuple{N,AbstractUnitRange}; kwargs...
) where {N}
  return oneelement(Bool, index, axes; kwargs...)
end

function oneelement(
  value, index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {N}
  return oneelement(value, index, Base.oneto.(size); kwargs...)
end
function oneelement(
  eltype::Type, index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...
) where {N}
  return oneelement(one(eltype), index, size; kwargs...)
end
function oneelement(index::NTuple{N,Int}, size::NTuple{N,Integer}; kwargs...) where {N}
  return oneelement(Bool, index, size; kwargs...)
end

function oneelement(value, ax_ind::Pair{<:AbstractUnitRange,Int}...; kwargs...)
  return oneelement(value, last.(ax_ind), first.(ax_ind); kwargs...)
end
function oneelement(eltype::Type, ax_ind::Pair{<:AbstractUnitRange,Int}...; kwargs...)
  return oneelement(one(eltype), ax_ind...; kwargs...)
end
function oneelement(ax_ind::Pair{<:AbstractUnitRange,Int}...; kwargs...)
  return oneelement(Bool, ax_ind...; kwargs...)
end

function oneelement(value; kwargs...)
  return oneelement(value, (), (); kwargs...)
end
function oneelement(eltype::Type; kwargs...)
  return oneelement(one(eltype); kwargs...)
end
function oneelement(; kwargs...)
  return oneelement(Bool; kwargs...)
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
  return a.getunstoredfun(a, I...)
end
function setstoredindex!(a::OneElementArray, value, I::Int...)
  return error("`OneElementArray` is immutable, you can't set elements.")
end
function setunstoredindex!(a::OneElementArray, value, I::Int...)
  return error("`OneElementArray` is immutable, you can't set elements.")
end
