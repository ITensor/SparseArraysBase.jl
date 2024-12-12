# This is to bring `ArrayLayouts.zero!` into the namespace
# since it is considered part of the sparse array interface.
using ArrayLayouts: zero!

# Minimal interface for `SparseArrayInterface`.
isstored(a::AbstractArray, I::Int...) = true
eachstoredindex(a::AbstractArray) = eachindex(a)
getstoredindex(a::AbstractArray, I::Int...) = getindex(a, I...)
function setstoredindex!(a::AbstractArray, value, I::Int...)
  setindex!(a, value, I...)
  return a
end
# TODO: Should this error by default if the value at the index
# is stored? It could be disabled with something analogous
# to `checkbounds`, like `checkstored`/`checkunstored`.
function setunstoredindex!(a::AbstractArray, value, I::Int...)
  setindex!(a, value, I...)
  return a
end

# TODO: Use `Base.to_indices`?
isstored(a::AbstractArray, I::CartesianIndex) = isstored(a, Tuple(I)...)
# TODO: Use `Base.to_indices`?
getstoredindex(a::AbstractArray, I::CartesianIndex) = getstoredindex(a, Tuple(I)...)
# TODO: Use `Base.to_indices`?
getunstoredindex(a::AbstractArray, I::CartesianIndex) = getunstoredindex(a, Tuple(I)...)
# TODO: Use `Base.to_indices`?
function setstoredindex!(a::AbstractArray, value, I::CartesianIndex)
  return setstoredindex!(a, value, Tuple(I)...)
end
# TODO: Use `Base.to_indices`?
function setunstoredindex!(a::AbstractArray, value, I::CartesianIndex)
  return setunstoredindex!(a, value, Tuple(I)...)
end

# Interface defaults.
# TODO: Have a fallback that handles element types
# that don't define `zero(::Type)`.
getunstoredindex(a::AbstractArray, I::Int...) = zero(eltype(a))

# Derived interface.
storedlength(a::AbstractArray) = length(storedvalues(a))
storedpairs(a::AbstractArray) = map(I -> I => getstoredindex(a, I), eachstoredindex(a))

to_vec(x) = vec(collect(x))
to_vec(x::AbstractArray) = vec(x)

# A view of the stored values of an array.
# Similar to: `@view a[collect(eachstoredindex(a))]`, but the issue
# with that is it returns a `SubArray` wrapping a sparse array, which
# is then interpreted as a sparse array so it can lead to recursion.
# Also, that involves extra logic for determining if the indices are
# stored or not, but we know the indices are stored so we can use
# `getstoredindex` and `setstoredindex!`.
# Most sparse arrays should overload `storedvalues` directly
# and avoid this wrapper since it adds extra indirection to
# access stored values.
struct StoredValues{T,A<:AbstractArray{T},I} <: AbstractVector{T}
  array::A
  storedindices::I
end
StoredValues(a::AbstractArray) = StoredValues(a, to_vec(eachstoredindex(a)))
Base.size(a::StoredValues) = size(a.storedindices)
Base.getindex(a::StoredValues, I::Int) = getstoredindex(a.array, a.storedindices[I])
function Base.setindex!(a::StoredValues, value, I::Int)
  return setstoredindex!(a.array, value, a.storedindices[I])
end

storedvalues(a::AbstractArray) = StoredValues(a)

function eachstoredindex(a1, a2, a_rest...)
  # TODO: Make this more customizable, say with a function
  # `combine/promote_storedindices(a1, a2)`.
  return union(eachstoredindex.((a1, a2, a_rest...))...)
end

using Derive: Derive, @derive, @interface, AbstractArrayInterface

# TODO: Add `ndims` type parameter.
# TODO: This isn't used to define interface functions right now.
# Currently, `@interface` expects an instance, probably it should take a
# type instead so fallback functions can use abstract types.
abstract type AbstractSparseArrayInterface <: AbstractArrayInterface end

# We restrict to `I::Vararg{Int,N}` to allow more general functions to handle trailing
# indices and linear indices.
@interface ::AbstractSparseArrayInterface function Base.getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  !isstored(a, I...) && return getunstoredindex(a, I...)
  return getstoredindex(a, I...)
end

# We restrict to `I::Vararg{Int,N}` to allow more general functions to handle trailing
# indices and linear indices.
@interface ::AbstractSparseArrayInterface function Base.setindex!(
  a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  if !isstored(a, I...)
    iszero(value) && return a
    setunstoredindex!(a, value, I...)
    return a
  end
  setstoredindex!(a, value, I...)
  return a
end

# TODO: This may need to be defined in `sparsearraydok.jl`, after `SparseArrayDOK`
# is defined. And/or define `default_type(::SparseArrayStyle, T::Type) = SparseArrayDOK{T}`.
@interface ::AbstractSparseArrayInterface function Base.similar(
  a::AbstractArray, T::Type, size::Tuple{Vararg{Int}}
)
  # TODO: Define `default_similartype` or something like that?
  return SparseArrayDOK{T}(size...)
end

@interface ::AbstractSparseArrayInterface function Base.map!(
  f, a_dest::AbstractArray, as::AbstractArray...
)
  # TODO: Define a function `preserves_unstored(a_dest, f, as...)`
  # to determine if a function preserves the stored values
  # of the destination sparse array.
  # The current code may be inefficient since it actually
  # accesses an unstored element, which in the case of a
  # sparse array of arrays can allocate an array.
  # Sparse arrays could be expected to define a cheap
  # unstored element allocator, for example
  # `get_prototypical_unstored(a::AbstractArray)`.
  I = first(eachindex(as...))
  preserves_unstored = iszero(f(map(a -> getunstoredindex(a, I), as)...))
  if !preserves_unstored
    # Doesn't preserve unstored values, loop over all elements.
    for I in eachindex(as...)
      a_dest[I] = map(f, map(a -> a[I], as)...)
    end
    return a_dest
  end
  # Define `eachstoredindex` promotion.
  for I in eachstoredindex(as...)
    a_dest[I] = f(map(a -> a[I], as)...)
  end
  return a_dest
end

# `f::typeof(norm)`, `op::typeof(max)` used by `norm`.
function reduce_init(f, op, as...)
  # TODO: Generalize this.
  @assert isone(length(as))
  a = only(as)
  ## TODO: Make this more efficient for block sparse
  ## arrays, in that case it allocates a block. Maybe
  ## it can use `FillArrays.Zeros`.
  return f(getunstoredindex(a, first(eachindex(a))))
end

@interface ::AbstractSparseArrayInterface function Base.mapreduce(
  f, op, as::AbstractArray...; init=reduce_init(f, op, as...), kwargs...
)
  # TODO: Generalize this.
  @assert isone(length(as))
  a = only(as)
  output = mapreduce(f, op, storedvalues(a); init, kwargs...)
  ## TODO: Bring this check back, or make the function more general.
  ## f_notstored = apply_notstored(f, a)
  ## @assert isequal(op(output, eltype(output)(f_notstored)), output)
  return output
end

abstract type AbstractSparseArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

@derive AbstractSparseArrayStyle AbstractArrayStyleOps

struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end

SparseArrayStyle{M}(::Val{N}) where {M,N} = SparseArrayStyle{N}()

@interface ::AbstractSparseArrayInterface function Broadcast.BroadcastStyle(type::Type)
  return SparseArrayStyle{ndims(type)}()
end

using ArrayLayouts: ArrayLayouts, MatMulMatAdd

abstract type AbstractSparseLayout <: ArrayLayouts.MemoryLayout end

function ArrayLayouts.sub_materialize(::AbstractSparseLayout, a::AbstractArray, axes::Tuple)
  a_dest = similar(a)
  a_dest .= a
  return a_dest
end

function mul_indices(I1::CartesianIndex{2}, I2::CartesianIndex{2})
  if I1[2] ≠ I2[1]
    return nothing
  end
  return CartesianIndex(I1[1], I2[2])
end

using LinearAlgebra: mul!
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

struct SparseLayout <: AbstractSparseLayout end

@interface ::AbstractSparseArrayInterface function ArrayLayouts.MemoryLayout(type::Type)
  return SparseLayout()
end
