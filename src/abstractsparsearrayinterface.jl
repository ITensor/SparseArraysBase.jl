using Base: @_propagate_inbounds_meta
using DerivableInterfaces:
    DerivableInterfaces, @derive, @interface, AbstractArrayInterface, zero!
using FillArrays: Zeros

function unstored end
function eachstoredindex end
function getstoredindex end
function getunstoredindex end
function isstored end
function setstoredindex! end
function setunstoredindex! end
function storedlength end
function storedpairs end
function storedvalues end

# Indicates that the array should be interpreted
# as the unstored values of a sparse array.
struct Unstored{T, N, P <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::P
end
Base.parent(a::Unstored) = a.parent
Base.size(a::Unstored) = size(parent(a))
Base.axes(a::Unstored) = axes(parent(a))

unstored(a::AbstractArray) = Zeros{eltype(a)}(axes(a))

function unstoredsimilar(a::AbstractArray, T::Type, ax::Tuple)
    return Zeros{T}(ax)
end
function unstoredsimilar(a::AbstractArray, ax::Tuple)
    return unstoredsimilar(a, eltype(a), ax)
end
function unstoredsimilar(a::AbstractArray, T::Type)
    return AbstractArray{T}(a)
end
unstoredsimilar(a::AbstractArray) = a

# Generic functionality for converting to a
# dense array, trying to preserve information
# about the array (such as which device it is on).
using TypeParameterAccessors: unspecify_type_parameters, unwrap_array, unwrap_array_type
function densetype(arraytype::Type{<:AbstractArray})
    return unspecify_type_parameters(unwrap_array_type(arraytype))
end
# TODO: Ideally this would be defined as `densetype(typeof(a))` but that is less general right now since `unwrap_array_type` is defined on fewer arrays, since it is based on `parentype` rather than `parent`.
function densetype(a::AbstractArray)
    return unspecify_type_parameters(typeof(unwrap_array(a)))
end
using GPUArraysCore: @allowscalar
function dense(a::AbstractArray)
    return @allowscalar convert(densetype(a), a)
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
    ) where {Interface <: AbstractSparseArrayInterface}
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
struct StoredValues{T, A <: AbstractArray{T}, I} <: AbstractVector{T}
    array::A
    storedindices::I
end
StoredValues(a::AbstractArray) = StoredValues(a, to_vec(eachstoredindex(a)))
Base.size(a::StoredValues) = size(a.storedindices)
@inline Base.getindex(a::StoredValues, I::Int) = getindex(a.array, a.storedindices[I])
@inline function Base.setindex!(a::StoredValues, value, I::Int)
    return setindex!(a.array, value, a.storedindices[I])
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
    @inbounds for I in eachstoredindex(a)
        a[I] = f(a[I])
    end
    return a
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
        f, op, as::AbstractArray...; init = reduce_init(f, op, as...), kwargs...
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

@derive (T = AbstractSparseArrayStyle,) begin
    Base.similar(::Broadcast.Broadcasted{<:T}, ::Type, ::Tuple)
    Base.copyto!(::AbstractArray, ::Broadcast.Broadcasted{<:T})
end

struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end

SparseArrayStyle{M}(::Val{N}) where {M, N} = SparseArrayStyle{N}()

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
        α::Number = true,
        β::Number = false,
    )
    mul!(a_dest, a1, a2, α, β)
    return a_dest
end

function default_mul!!(
        a_dest::Number, a1::Number, a2::Number, α::Number = true, β::Number = false
    )
    return a1 * a2 * α + a_dest * β
end

# a1 * a2 * α + a_dest * β
function sparse_mul!(
        a_dest::AbstractArray,
        a1::AbstractArray,
        a2::AbstractArray,
        α::Number = true,
        β::Number = false;
        (mul!!) = (default_mul!!),
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
        m::MatMulMatAdd{<:AbstractSparseLayout, <:AbstractSparseLayout, <:AbstractSparseLayout}
    )
    sparse_mul!(m.C, m.A, m.B, m.α, m.β)
    return m.C
end

struct SparseLayout <: AbstractSparseLayout end

@interface ::AbstractSparseArrayInterface function ArrayLayouts.MemoryLayout(type::Type)
    return SparseLayout()
end
