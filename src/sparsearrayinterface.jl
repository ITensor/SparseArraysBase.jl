using FunctionImplementations: FunctionImplementations

struct SparseArrayStyle <: AbstractSparseArrayStyle end

# Convenient shorthand to refer to the sparse interface.
# Can turn a function into a sparse function with the syntax `sparse_style(f)`,
# i.e. `sparse_style(map)(x -> 2x, randn(2, 2))` while use the sparse
# version of `map`.
const sparse_style = SparseArrayStyle()

## # Fix ambiguity error.
## function DerivableInterfaces.combine_interface_rule(
##         ::SparseArrayInterface{N}, ::SparseArrayInterface{N}
##     ) where {N}
##     return SparseArrayInterface{N}()
## end
## function DerivableInterfaces.combine_interface_rule(
##         ::SparseArrayInterface, ::SparseArrayInterface
##     )
##     return SparseArrayInterface()
## end
## function DerivableInterfaces.combine_interface_rule(
##         interface1::SparseArrayInterface, interface2::AbstractSparseArrayInterface
##     )
##     return interface1
## end
## function DerivableInterfaces.combine_interface_rule(
##         interface1::AbstractSparseArrayInterface, interface2::SparseArrayInterface
##     )
##     return interface2
## end

## FunctionImplementations.Style(::Type{<:AbstractSparseArrayStyle}) = SparseArrayStyle()

## using FunctionImplementations: zero!
##
## # `zero!` isn't defined in `Base`, but it is defined in `ArrayLayouts`
## # and is useful for sparse array logic, since it can be used to empty
## # the sparse array storage.
## # We use a single function definition to minimize method ambiguities.
## const zero!_sparse = sparse_style(zero!)
## function zero!_sparse(a::AbstractArray)
##     # More generally, this codepath could be taking if `zero(eltype(a))`
##     # is defined and the elements are immutable.
##     f = eltype(a) <: Number ? Returns(zero(eltype(a))) : zero!
##     @inbounds for I in eachstoredindex(a)
##         a[I] = f(a[I])
##     end
##     return a
## end

using FunctionImplementations: FunctionImplementations, zero!

# `zero!` isn't defined in `Base`, but it is defined in `ArrayLayouts`
# and is useful for sparse array logic, since it can be used to empty
# the sparse array storage.
# We use a single function definition to minimize method ambiguities.
const zero!_sparse = sparse_style(zero!)
function zero!_sparse(a::AbstractArray)
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

const mapreduce_sparse = sparse_style(mapreduce)
function mapreduce_sparse(
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

module Broadcast

    abstract type AbstractSparseArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

    ## @derive (T = AbstractSparseArrayStyle,) begin
    ##     Base.similar(::Broadcast.Broadcasted{<:T}, ::Type, ::Tuple)
    ##     Base.copyto!(::AbstractArray, ::Broadcast.Broadcasted{<:T})
    ## end

    struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end

    SparseArrayStyle{M}(::Val{N}) where {M, N} = SparseArrayStyle{N}()

    # TODO: Don't make this a `sparse_style` function.
    ## using ..SparseArraysBase: sparse_style
    ## const BroadcastStyle_sparse = sparse_style(Base.Broadcast.BroadcastStyle)
    ## function Base.Broadcast.BroadcastStyle(type::Type{<:AnyAbstractSparseArray})
    ##     return SparseArrayStyle{ndims(type)}()
    ## end

end # module Broadcast

## # TODO: Don't make this a `sparse_style` function.
struct SparseLayout <: AbstractSparseLayout end
## const MemoryLayout_sparse = sparse_style(ArrayLayouts.MemoryLayout)
## ArrayLayouts.MemoryLayout(type::Type{<:AnyAbstractSparseArray}) = SparseLayout()
