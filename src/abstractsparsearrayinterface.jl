# Minimal interface for `SparseArrayInterface`.
# TODO: Define default definitions for these based
# on the dense case.
storedvalues(a) = error()
isstored(a, I::Int...) = error()
eachstoredindex(a) = error()
getstoredindex(a, I::Int...) = error()
setstoredindex!(a, value, I::Int...) = error()
setunstoredindex!(a, value, I::Int...) = error()

# Interface defaults.
# TODO: Have a fallback that handles element types
# that don't define `zero(::Type)`.
getunstoredindex(a, I::Int...) = zero(eltype(a))

# Derived interface.
storedlength(a) = length(storedvalues(a))
storedpairs(a) = map(I -> I => getstoredindex(a, I), eachstoredindex(a))

function eachstoredindex(a1, a2, a_rest...)
  # TODO: Make this more customizable, say with a function
  # `combine/promote_storedindices(a1, a2)`.
  return union(eachstoredindex.((a1, a2, a_rest...))...)
end

using Derive: Derive, @interface, AbstractArrayInterface

# TODO: Add `ndims` type parameter.
# TODO: This isn't used to define interface functions right now.
# Currently, `@interface` expects an instance, probably it should take a
# type instead so fallback functions can use abstract types.
abstract type AbstractSparseArrayInterface <: AbstractArrayInterface end

# TODO: Use `ArrayLayouts.layout_getindex`, `ArrayLayouts.sub_materialize`
# to handle slicing (implemented by copying SubArray).
@interface AbstractSparseArrayInterface function Base.getindex(a, I::Int...)
  !isstored(a, I...) && return getunstoredindex(a, I...)
  return getstoredindex(a, I...)
end

@interface AbstractSparseArrayInterface function Base.setindex!(a, value, I::Int...)
  iszero(value) && return a
  if !isstored(a, I...)
    setunstoredindex!(a, value, I...)
    return a
  end
  setstoredindex!(a, value, I...)
  return a
end

# TODO: This may need to be defined in `sparsearraydok.jl`, after `SparseArrayDOK`
# is defined. And/or define `default_type(::SparseArrayStyle, T::Type) = SparseArrayDOK{T}`.
@interface AbstractSparseArrayInterface function Base.similar(
  a, T::Type, size::Tuple{Vararg{Int}}
)
  # TODO: Define `default_similartype` or something like that?
  return SparseArrayDOK{T}(size...)
end

@interface AbstractSparseArrayInterface function Base.map!(f, dest, as...)
  # Check `f` preserves zeros.
  # Define as `map_stored!`.
  # Define `eachstoredindex` promotion.
  for I in eachstoredindex(as...)
    dest[I] = f(map(a -> a[I], as)...)
  end
  return dest
end

# TODO: Make this a subtype of `Derive.AbstractArrayStyle{N}` instead.
using Derive: Derive
abstract type AbstractSparseArrayStyle{N} <: Derive.AbstractArrayStyle{N} end

struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end

SparseArrayStyle{M}(::Val{N}) where {M,N} = SparseArrayStyle{N}()

@interface AbstractSparseArrayInterface function Broadcast.BroadcastStyle(type::Type)
  return SparseArrayStyle{ndims(type)}()
end

using ArrayLayouts: ArrayLayouts, MatMulMatAdd

abstract type AbstractSparseLayout <: ArrayLayouts.MemoryLayout end

struct SparseLayout <: AbstractSparseLayout end

@interface AbstractSparseArrayInterface function ArrayLayouts.MemoryLayout(type::Type)
  return SparseLayout()
end

function mul_indices(I1::CartesianIndex{2}, I2::CartesianIndex{2})
  if I1[2] ≠ I2[1]
    return nothing
  end
  return CartesianIndex(I1[1], I2[2])
end

function default_mul!!(
  a_dest::AbstractMatrix,
  a1::AbstractMatrix,
  a2::AbstractMatrix,
  α::Number=true,
  β::Number=false,
)
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end

function default_mul!!(
  a_dest::Number, a1::Number, a2::Number, α::Number=true, β::Number=false
)
  return a1 * a2 * α + a_dest * β
end

# a1 * a2 * α + a_dest * β
function sparse_mul!(
  a_dest::AbstractArray,
  a1::AbstractArray,
  a2::AbstractArray,
  α::Number=true,
  β::Number=false;
  (mul!!)=(default_mul!!),
)
  for I1 in eachstoredindex(a1)
    for I2 in eachstoredindex(a2)
      I_dest = mul_indices(I1, I2)
      if !isnothing(I_dest)
        a_dest[I_dest] = mul!!(a_dest[I_dest], a1[I1], a2[I2], α, β)
      end
    end
  end
  return a_dest
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{<:AbstractSparseLayout,<:AbstractSparseLayout,<:AbstractSparseLayout}
)
  sparse_mul!(m.C, m.A, m.B, m.α, m.β)
  return m.C
end
