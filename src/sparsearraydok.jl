using Accessors: @set
using DerivableInterfaces: DerivableInterfaces, @interface, interface, zero!
using Dictionaries: Dictionary, IndexError, set!

const DOKStorage{T, N} = Dictionary{CartesianIndex{N}, T}

function _SparseArrayDOK end

"""
    SparseArrayDOK{T,N,F} <: AbstractSparseArray{T,N}

`N`-dimensional sparse dictionary-of-keys (DOK) array with elements of type `T`,
with a specified background of unstored values `unstored` of the size of the array.
"""
struct SparseArrayDOK{T, N, Unstored <: AbstractArray{T, N}} <: AbstractSparseArray{T, N}
    storage::DOKStorage{T, N}
    unstored::Unstored
    global @inline function _SparseArrayDOK(
            storage::DOKStorage{T, N}, unstored::Unstored
        ) where {T, N, Unstored <: AbstractArray{T, N}}
        return new{T, N, Unstored}(storage, unstored)
    end
end

unstored(a::SparseArrayDOK) = a.unstored
Base.size(a::SparseArrayDOK) = size(unstored(a))
Base.axes(a::SparseArrayDOK) = axes(unstored(a))

function SparseArrayDOK{T, N}(::UndefInitializer, a::Unstored) where {T, N}
    storage = DOKStorage{T, N}()
    return _SparseArrayDOK(storage, parent(a))
end
function SparseArrayDOK{T}(::UndefInitializer, a::Unstored) where {T}
    return SparseArrayDOK{T, ndims(a)}(undef, a)
end
function SparseArrayDOK{<:Any, N}(::UndefInitializer, a::Unstored) where {N}
    return SparseArrayDOK{eltype(a), N}(a)
end
function SparseArrayDOK(::UndefInitializer, a::Unstored)
    return SparseArrayDOK{eltype(a), ndims(a)}(undef, a)
end

# Constructors
# ------------
"""
    SparseArrayDOK{T}(undef, dims...)
    SparseArrayDOK{T,N}(undef, dims...)

Construct an uninitialized `N`-dimensional [`SparseArrayDOK`](@ref) containing
elements of type `T`. `N` can either be supplied explicitly, or be determined by
the length or number of `dims`.
"""
SparseArrayDOK{T, N}(::UndefInitializer, dims...)

function SparseArrayDOK{T, N}(::UndefInitializer, ax::Tuple{Vararg{Any, N}}) where {T, N}
    return SparseArrayDOK{T, N}(undef, Unstored(Zeros{T}(ax)))
end
function SparseArrayDOK{T}(::UndefInitializer, ax::Tuple{Vararg{Any, N}}) where {T, N}
    return SparseArrayDOK{T, N}(undef, ax)
end
function SparseArrayDOK{T, N}(::UndefInitializer, ax::Vararg{Int, N}) where {T, N}
    return SparseArrayDOK{T, N}(undef, ax)
end
function SparseArrayDOK{T}(::UndefInitializer, ax::Vararg{Any, N}) where {T, N}
    return SparseArrayDOK{T, N}(undef, ax)
end

using DerivableInterfaces: DerivableInterfaces
# This defines the destination type of various operations in DerivableInterfaces.jl.
function Base.similar(::AbstractSparseArrayInterface, T::Type, ax::Tuple)
    return similar(SparseArrayDOK{T}, ax)
end

using DerivableInterfaces: @array_aliases
# Define `SparseMatrixDOK`, `AnySparseArrayDOK`, etc.
@array_aliases SparseArrayDOK

storage(a::SparseArrayDOK) = a.storage

storedvalues(a::SparseArrayDOK) = values(storage(a))
@inline function isstored(a::SparseArrayDOK{<:Any, N}, I::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(a, I...)
    return haskey(storage(a), CartesianIndex(I))
end
function eachstoredindex(::IndexCartesian, a::SparseArrayDOK)
    return keys(storage(a))
end
@inline function getstoredindex(a::SparseArrayDOK{<:Any, N}, I::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(a, I...)
    return storage(a)[CartesianIndex(I)]
end
@inline function getunstoredindex(a::SparseArrayDOK{<:Any, N}, I::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(a, I...)
    return unstored(a)[I...]
end
@inline function setstoredindex!(
        a::SparseArrayDOK{<:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    # `isstored` includes a boundscheck as well
    @boundscheck isstored(a, I...) ||
        throw(IndexError(lazy"key $(CartesianIndex(I...)) not found"))
    # TODO: If `iszero(value)`, unstore the index.
    storage(a)[CartesianIndex(I)] = value
    return a
end
@inline function setunstoredindex!(
        a::SparseArrayDOK{<:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    @boundscheck checkbounds(a, I...)
    insert!(storage(a), CartesianIndex(I), value)
    return a
end

# Optional, but faster than the default.
storedpairs(a::SparseArrayDOK) = pairs(storage(a))

# TODO: Also handle wrappers.
function DerivableInterfaces.zero!(a::SparseArrayDOK)
    empty!(storage(a))
    return a
end
function ArrayLayouts.zero!(a::SparseArrayDOK)
    return zero!(a)
end
