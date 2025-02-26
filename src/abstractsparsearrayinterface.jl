using DerivableInterfaces: DerivableInterfaces, @derive, @interface, AbstractArrayInterface

# This is to bring `ArrayLayouts.zero!` into the namespace
# since it is considered part of the sparse array interface.

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

"""
    AbstractSparseArrayInterface <: AbstractArrayInterface

Abstract supertype for any interface associated with sparse array types.
"""
abstract type AbstractSparseArrayInterface <: AbstractArrayInterface end

"""
    SparseArrayInterface <: AbstractSparseArrayInterface

Interface for array operations that are centered around sparse storage types, typically assuming
fast `O(1)` random access/insertion, but slower sequential access.
"""
struct SparseArrayInterface <: AbstractSparseArrayInterface end

# by default, this interface is stronger than other interfaces (is this fair?)

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

# Fix ambiguity error.
function DerivableInterfaces.combine_interface_rule(
  ::SparseArrayInterface, ::SparseArrayInterface
)
  return SparseArrayInterface()
end
function DerivableInterfaces.combine_interface_rule(
  interface1::SparseArrayInterface, interface2::AbstractSparseArrayInterface
)
  return interface1
end
function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractSparseArrayInterface, interface2::SparseArrayInterface
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
# struct StoredValues{T,A<:AbstractArray{T},I} <: AbstractVector{T}
#   array::A
#   storedindices::I
# end
# StoredValues(a::AbstractArray) = StoredValues(a, to_vec(eachstoredindex(a)))
# Base.size(a::StoredValues) = size(a.storedindices)
# Base.getindex(a::StoredValues, I::Int) = getstoredindex(a.array, a.storedindices[I])
# function Base.setindex!(a::StoredValues, value, I::Int)
#   return setstoredindex!(a.array, value, a.storedindices[I])
# end
#
# @interface ::AbstractSparseArrayInterface storedvalues(a::AbstractArray) = StoredValues(a)

# TODO: This may need to be defined in `sparsearraydok.jl`, after `SparseArrayDOK`
# is defined. And/or define `default_type(::SparseArrayStyle, T::Type) = SparseArrayDOK{T}`.
@interface ::AbstractSparseArrayInterface function Base.similar(
  a::AbstractArray, T::Type, size::Tuple{Vararg{Int}}
)
  # TODO: Define `default_similartype` or something like that?
  return SparseArrayDOK{T}(size...)
end

using ArrayLayouts: ArrayLayouts, zero!

# `zero!` isn't defined in `Base`, but it is defined in `ArrayLayouts`
# and is useful for sparse array logic, since it can be used to empty
# the sparse array storage.
# We use a single function definition to minimize method ambiguities.
@interface interface::AbstractSparseArrayInterface function ArrayLayouts.zero!(
  a::AbstractArray
)
  # More generally, this codepath could be taking if `zero(eltype(a))`
  # is defined and the elements are immutable.
  f = eltype(a) <: Number ? Returns(zero(eltype(a))) : zero!
  return @interface interface map_stored!(f, a, a)
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

@derive (T=AbstractSparseArrayStyle,) begin
  Base.similar(::Broadcast.Broadcasted{<:T}, ::Type, ::Tuple)
  Base.copyto!(::AbstractArray, ::Broadcast.Broadcasted{<:T})
end

struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end

SparseArrayStyle{M}(::Val{N}) where {M,N} = SparseArrayStyle{N}()

DerivableInterfaces.interface(::Type{<:AbstractSparseArrayStyle}) = SparseArrayInterface()

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

using LinearAlgebra: LinearAlgebra, mul!

# a1 * a2 * α + a_dest * β
@interface ::AbstractSparseArrayInterface function LinearAlgebra.mul!(
  C::AbstractArray, A::AbstractArray, B::AbstractArray, α::Number, β::Number
)
  a_dest .*= β
  β′ = one(Bool)
  for I1 in eachstoredindex(a1)
    for I2 in eachstoredindex(a2)
      I_dest = mul_indices(I1, I2)
      if !isnothing(I_dest)
        a_dest[I_dest] = mul!(a_dest[I_dest], a1[I1], a2[I2], α, β′)
      end
    end
  end
  return a_dest
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{<:AbstractSparseLayout,<:AbstractSparseLayout,<:AbstractSparseLayout}
)
  @interface SparseArrayInterface() mul!(m.C, m.A, m.B, m.α, m.β)
  return m.C
end

struct SparseLayout <: AbstractSparseLayout end

@interface ::AbstractSparseArrayInterface function ArrayLayouts.MemoryLayout(type::Type)
  return SparseLayout()
end
