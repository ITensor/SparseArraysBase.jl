using FunctionImplementations: FunctionImplementations

struct SparseArrayImplementationStyle <: AbstractSparseArrayImplementationStyle end

# Convenient shorthand to refer to the sparse style.
# Can turn a function into a sparse function with the syntax `sparse_style(f)`,
# i.e. `sparse_style(map)(x -> 2x, randn(2, 2))` while use the sparse
# version of `map`.
const sparse_style = SparseArrayImplementationStyle()

const fill!_sparse = sparse_style(fill!)
function fill!_sparse(a::AbstractArray, value)
    return map!(Returns(value), a, a)
end

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

const zero_sparse = sparse_style(zero)
# Specialized version of `Base.zero` written in terms of `zero!`.
# This is friendlier for sparse arrays since `zero!` makes it easier
# to handle the logic of dropping all elements of the sparse array when possible.
# We use a single function definition to minimize method ambiguities.
function zero_sparse(a::AbstractArray)
    # More generally, the first codepath could be taking if `zero(eltype(a))`
    # is defined and the elements are immutable.
    if eltype(a) <: Number
        return zero!(similar(a))
    end
    return map(zero, a)
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

# This is defined in this way so we can rely on the Broadcast logic
# for determining the destination of the operation (element type, shape, etc.).
const map_sparse = sparse_style(map)
function map_sparse(f, as::AbstractArray...)
    # Broadcasting is used here to determine the destination array but that
    # could be done manually here.
    return f.(as...)
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

abstract type AbstractSparseArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end
SparseArrayStyle{M}(::Val{N}) where {M, N} = SparseArrayStyle{N}()

using MapBroadcast: Mapped
# TODO: Look into `SparseArrays.capturescalars`:
# https://github.com/JuliaSparse/SparseArrays.jl/blob/1beb0e4a4618b0399907b0000c43d9f66d34accc/src/higherorderfns.jl#L1092-L1102
function Base.copyto!(
        a_dest::AbstractArray, bc::Base.Broadcast.Broadcasted{<:SparseArrayStyle}
    )
    m = Mapped(bc)
    map!(m.f, a_dest, m.args...)
    return a_dest
end

function Base.similar(
        bc::Base.Broadcast.Broadcasted{<:SparseArrayStyle}, elt::Type, ax
    )
    return similar(SparseArrayDOK{elt}, ax)
end

using ArrayLayouts: ArrayLayouts
const mul!_sparse = sparse_style(mul!)
function mul!_sparse(
        a_dest::AbstractVecOrMat, a1::AbstractVecOrMat, a2::AbstractVecOrMat, α::Number,
        β::Number
    )
    return ArrayLayouts.mul!(a_dest, a1, a2, α, β)
end

struct SparseLayout <: AbstractSparseLayout end
