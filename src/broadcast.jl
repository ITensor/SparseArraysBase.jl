using Base.Broadcast: Broadcasted
using MapBroadcast: Mapped

abstract type AbstractSparseArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end
SparseArrayStyle{M}(::Val{N}) where {M,N} = SparseArrayStyle{N}()

DerivableInterfaces.interface(::Type{<:AbstractSparseArrayStyle}) = SparseArrayInterface()

@derive (T=AbstractSparseArrayStyle,) begin
  Base.similar(::Broadcasted{<:T}, ::Type, ::Tuple)
  Base.copyto!(::AbstractArray, ::Broadcasted{<:T})
end

@interface ::AbstractSparseArrayInterface function Broadcast.BroadcastStyle(type::Type)
  return SparseArrayStyle{ndims(type)}()
end

@interface I::AbstractSparseArrayInterface function Base.similar(
  ::Broadcasted, ::Type{T}, ax
) where {T}
  return similar(I, T, ax)
end

@interface I::AbstractSparseArrayInterface function Base.copyto!(
  C::AbstractArray, bc::Broadcasted
)
  m = Mapped(bc)
  return @interface I map!(m.f, C, m.args...)
end
