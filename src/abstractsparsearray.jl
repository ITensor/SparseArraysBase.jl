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
  Base.getindex(::T, ::Any...)
  Base.getindex(::T, ::Int...)
  Base.setindex!(::T, ::Any, ::Any...)
  Base.setindex!(::T, ::Any, ::Int...)
  Base.similar(::T, ::Type, ::Tuple{Vararg{Int}})
  Base.similar(::T, ::Type, ::Tuple{Base.OneTo,Vararg{Base.OneTo}})
  Base.copy(::T)
  Base.copy!(::AbstractArray, ::T)
  Base.copyto!(::AbstractArray, ::T)
  Base.map(::Any, ::T...)
  Base.map!(::Any, ::AbstractArray, ::T...)
  Base.mapreduce(::Any, ::Any, ::T...; kwargs...)
  Base.reduce(::Any, ::T...; kwargs...)
  Base.all(::Function, ::T)
  Base.all(::T)
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
  Base.show(::IO, ::MIMEtextplain, ::T)
end

function Base.replace_in_print_matrix(
  A::AnyAbstractSparseArray{<:Any,2}, i::Integer, j::Integer, s::AbstractString
)
  return isstored(A, CartesianIndex(i, j)) ? s : Base.replace_with_centered_mark(s)
end
