# zero-preserving Traits
# ----------------------
"""
    abstract type ZeroPreserving <: Function end

Holy Trait to indicate how a function interacts with abstract zero values:

- `StrongPreserving` : output is guaranteed to be zero if **any** input is.
- `WeakPreserving` : output is guaranteed to be zero if **all** inputs are.
- `NonPreserving` : no guarantees on output.

To attempt to automatically determine this, either `ZeroPreserving(f, A::AbstractArray...)` or
`ZeroPreserving(f, T::Type...)` can be used/overloaded.

!!! warning
    incorrectly registering a function to be zero-preserving will lead to silently wrong results.
"""
abstract type ZeroPreserving <: Function end

struct StrongPreserving{F} <: ZeroPreserving
    f::F
end
struct WeakPreserving{F} <: ZeroPreserving
    f::F
end
struct NonPreserving{F} <: ZeroPreserving
    f::F
end

# Backport: remove in 1.12
@static if !isdefined(Base, :haszero)
    _haszero(T::Type) = false
    _haszero(::Type{<:Number}) = true
else
    _haszero = Base.haszero
end

# warning: cannot automatically detect WeakPreserving since this would mean checking all values
function ZeroPreserving(f, A::AbstractArray, Bs::AbstractArray...)
    return ZeroPreserving(f, eltype(A), eltype.(Bs)...)
end
# TODO: the following might not properly specialize on the types
# TODO: non-concrete element types
function ZeroPreserving(f, T::Type, Ts::Type...)
    if all(_haszero, (T, Ts...))
        return iszero(f(zero(T), zero.(Ts)...)) ? WeakPreserving(f) : NonPreserving(f)
    else
        return NonPreserving(f)
    end
end
ZeroPreserving(f::ZeroPreserving, T::Type, Ts::Type...) = f

for F in (:(typeof(+)), :(typeof(-)), :(typeof(identity)))
    @eval begin
        ZeroPreserving(f::$F, ::Type, ::Type...) = WeakPreserving(f)
    end
end

using MapBroadcast: MapFunction
for F in (:(typeof(*)), :(MapFunction{typeof(*)}))
    @eval begin
        function ZeroPreserving(f::$F, ::Type, ::Type...)
            return StrongPreserving(f)
        end
    end
end

# map(!)
# ------
const map_sparse = sparse_style(map)
function map_sparse(
        f, A::AbstractArray, Bs::AbstractArray...
    )
    f_pres = ZeroPreserving(f, A, Bs...)
    return map_sparse(f_pres, A, Bs...)
end

# This isn't an overload of `Base.map` since that leads to ambiguity errors.
function map_sparse(f::ZeroPreserving, A::AbstractArray, Bs::AbstractArray...)
    T = Base.Broadcast.combine_eltypes(f.f, (A, Bs...))
    C = similar(A, T)
    # TODO: Instead use:
    # ```julia
    # U = map(f.f, map(unstored, (A, Bs...))...)
    # C = similar(A, Unstored(U))
    # ```
    # though right now `map` doesn't preserve `Zeros` or `BlockZeros`.
    return map!_sparse(f, C, A, Bs...)
end

const map!_sparse = sparse_style(map!)
function map!_sparse(
        f, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...
    )
    f_pres = ZeroPreserving(f, A, Bs...)
    return map!_sparse(f_pres, C, A, Bs...)
end

# This isn't an overload of `Base.map!` since that leads to ambiguity errors.
function map!_sparse(
        f::ZeroPreserving, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...
    )
    checkshape(C, A, Bs...)
    unaliased = map(Base.Fix1(Base.unalias, C), (A, Bs...))
    if f isa StrongPreserving
        style = IndexStyle(C, unaliased...)
        inds = intersect(eachstoredindex.(Ref(style), unaliased)...)
        zero!(C)
    elseif f isa WeakPreserving
        style = IndexStyle(C, unaliased...)
        inds = union(eachstoredindex.(Ref(style), unaliased)...)
        zero!(C)
    elseif f isa NonPreserving
        inds = eachindex(C, unaliased...)
    else
        error(lazy"unknown zero-preserving type $(typeof(f))")
    end
    @inbounds for I in inds
        C[I] = f.f(ith_all(I, unaliased)...)
    end
    return C
end

# Derived functions
# -----------------
const copyto!_sparse = sparse_style(copyto!)
function copyto!_sparse(
        dest::AbstractArray, src::AbstractArray
    )
    map!_sparse(identity, dest, src)
    return dest
end

const permutedims!_sparse = sparse_style(permutedims!)
function permutedims!_sparse(
        a_dest::AbstractArray, a_src::AbstractArray, perm
    )
    return map!(identity, a_dest, PermutedDimsArray(a_src, perm))
end

# Only map the stored values of the inputs.
function map_stored! end

const map_stored!_sparse = sparse_style(map_stored!)
function map_stored!_sparse(
        f, a_dest::AbstractArray, as::AbstractArray...
    )
    map!_sparse(WeakPreserving(f), a_dest, as...)
    return a_dest
end

# Only map all values, not just the stored ones.
function map_all! end

const map_all!_sparse = sparse_style(map_all!)
function map_all!_sparse(
        f, a_dest::AbstractArray, as::AbstractArray...
    )
    map!_sparse(NonPreserving(f), a_dest, as...)
    return a_dest
end

# TODO: Generalize to multiple inputs.
const reduce_sparse = sparse_style(reduce)
function reduce_sparse(f, a::AbstractArray; kwargs...)
    return mapreduce(identity, f, a; kwargs...)
end

const all_sparse = sparse_style(all)
function all_sparse(a::AbstractArray)
    return reduce(&, a; init = true)
end
function all_sparse(f::Function, a::AbstractArray)
    return mapreduce(f, &, a; init = true)
end

const isreal_sparse = sparse_style(isreal)
isreal_sparse(a::AbstractArray) = all(isreal, a)

const iszero_sparse = sparse_style(iszero)
iszero_sparse(a::AbstractArray) = all(iszero, a)

# Utility functions
# -----------------
# shape check similar to checkbounds
checkshape(::Type{Bool}, A::AbstractArray) = true
checkshape(::Type{Bool}, A::AbstractArray, B::AbstractArray) = size(A) == size(B)
function checkshape(::Type{Bool}, A::AbstractArray, Bs::AbstractArray...)
    return allequal(size, (A, Bs...))
end

function checkshape(A::AbstractArray, Bs::AbstractArray...)
    return checkshape(Bool, A, Bs...) ||
        throw(DimensionMismatch("argument shapes must match"))
end

@inline ith_all(i, ::Tuple{}) = ()
function ith_all(i, as)
    @_propagate_inbounds_meta
    return (as[1][i], ith_all(i, Base.tail(as))...)
end
