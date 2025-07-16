using Base: @_propagate_inbounds_meta
using DerivableInterfaces:
  DerivableInterfaces, @derive, @interface, AbstractArrayInterface, zero!

function eachstoredindex end
function getstoredindex end
function getunstoredindex end
function isstored end
function setstoredindex! end
function setunstoredindex! end
function storedlength end
function storedpairs end
function storedvalues end

# Replace the function for accessing
# unstored values.
function set_getunstoredindex end

# Generic functionality for converting to a
# dense array, trying to preserve information
# about the array (such as which device it is on).
# TODO: Maybe call `densecopy`?
# TODO: Make sure this actually preserves the device,
# maybe use `TypeParameterAccessors.unwrap_array_type`.
# TODO: Turn into an `@interface` function.
function densearray(a::AbstractArray)
  # TODO: `set_ndims(unwrap_array_type(a), ndims(a))(a)`
  # Maybe define `densetype(a) = set_ndims(unwrap_array_type(a), ndims(a))`.
  # Or could use `unspecify_parameters(unwrap_array_type(a))(a)`.
  return Array(a)
end

# Minimal interface for `SparseArrayInterface`.
# Fallbacks for dense/non-sparse arrays.

# TODO: Add `ndims` type parameter, like `Base.Broadcast.AbstractArrayStyle`.
# TODO: This isn't used to define interface functions right now.
# Currently, `@interface` expects an instance, probably it should take a
# type instead so fallback functions can use abstract types.
abstract type AbstractSparseArrayInterface{N} <: AbstractArrayInterface{N} end

function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractSparseArrayInterface, interface2::AbstractSparseArrayInterface
)
  return error("Rule not defined.")
end
function DerivableInterfaces.combine_interface_rule(
  interface1::Interface, interface2::Interface
) where {Interface<:AbstractSparseArrayInterface}
  return interface1
end
function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractSparseArrayInterface, interface2::AbstractArrayInterface
)
  return interface1
end
function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractArrayInterface, interface2::AbstractSparseArrayInterface
)
  return interface2
end

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
@inline Base.getindex(a::StoredValues, I::Int) = getindex(a.array, a.storedindices[I])
@inline function Base.setindex!(a::StoredValues, value, I::Int)
  return setindex!(a.array, value, a.storedindices[I])
end

# TODO: This may need to be defined in `sparsearraydok.jl`, after `SparseArrayDOK`
# is defined. And/or define `default_type(::SparseArrayStyle, T::Type) = SparseArrayDOK{T}`.
@interface ::AbstractSparseArrayInterface function Base.similar(
  a::AbstractArray, T::Type, size::Tuple{Vararg{Int}}
)
  # TODO: Define `default_similartype` or something like that?
  return SparseArrayDOK{T}(undef, size)
end

# map over a specified subset of indices of the inputs.
function map_indices! end

@interface interface::AbstractArrayInterface function map_indices!(
  indices, f, a_dest::AbstractArray, as::AbstractArray...
)
  for I in indices
    a_dest[I] = f(map(a -> a[I], as)...)
  end
  return a_dest
end

# Only map the stored values of the inputs.
function map_stored! end

@interface interface::AbstractArrayInterface function map_stored!(
  f, a_dest::AbstractArray, as::AbstractArray...
)
  @interface interface map_indices!(eachstoredindex(as...), f, a_dest, as...)
  return a_dest
end

# Only map all values, not just the stored ones.
function map_all! end

@interface interface::AbstractArrayInterface function map_all!(
  f, a_dest::AbstractArray, as::AbstractArray...
)
  @interface interface map_indices!(eachindex(as...), f, a_dest, as...)
  return a_dest
end

using DerivableInterfaces: DerivableInterfaces, zero!

# `zero!` isn't defined in `Base`, but it is defined in `ArrayLayouts`
# and is useful for sparse array logic, since it can be used to empty
# the sparse array storage.
# We use a single function definition to minimize method ambiguities.
@interface interface::AbstractSparseArrayInterface function DerivableInterfaces.zero!(
  a::AbstractArray
)
  # More generally, this codepath could be taking if `zero(eltype(a))`
  # is defined and the elements are immutable.
  f = eltype(a) <: Number ? Returns(zero(eltype(a))) : zero!
  return @interface interface map_stored!(f, a, a)
end

# Determines if a function preserves the stored values
# of the destination sparse array.
# The current code may be inefficient since it actually
# accesses an unstored element, which in the case of a
# sparse array of arrays can allocate an array.
# Sparse arrays could be expected to define a cheap
# unstored element allocator, for example
# `get_prototypical_unstored(a::AbstractArray)`.
function preserves_unstored(f, a_dest::AbstractArray, as::AbstractArray...)
  I = first(eachindex(as...))
  return iszero(f(map(a -> getunstoredindex(a, I), as)...))
end

# @interface interface::AbstractSparseArrayInterface function Base.map!(
#   f, a_dest::AbstractArray, as::AbstractArray...
# )
#   isempty(a_dest) && return a_dest # special case to avoid trying to access empty array
#   indices = if !preserves_unstored(f, a_dest, as...)
#     eachindex(a_dest)
#   elseif any(a -> a_dest !== a, as)
#     as = map(a -> Base.unalias(a_dest, a), as)
#     @interface interface zero!(a_dest)
#     eachstoredindex(as...)
#   else
#     eachstoredindex(a_dest)
#   end
#   @interface interface map_indices!(indices, f, a_dest, as...)
#   return a_dest
# end

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
  a_dest .*= β
  β′ = one(Bool)
  for I1 in eachstoredindex(a1)
    for I2 in eachstoredindex(a2)
      I_dest = mul_indices(I1, I2)
      if !isnothing(I_dest)
        if isstored(a_dest, I_dest)
          a_dest[I_dest] = mul!!(a_dest[I_dest], a1[I1], a2[I2], α, β′)
        else
          a_dest[I_dest] = a1[I1] * a2[I2] * α
        end
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
