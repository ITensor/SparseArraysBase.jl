using Dictionaries: AbstractDictionary

abstract type AbstractSparseArray{T, N} <: AbstractArray{T, N} end
const AbstractSparseVector{T} = AbstractSparseArray{T, 1}
const AbstractSparseMatrix{T} = AbstractSparseArray{T, 2}

using Adapt: WrappedArray
using LinearAlgebra: Adjoint, Transpose
const WrappedAbstractSparseArray{T, N} =
    Union{
    WrappedArray{T, N, AbstractSparseArray, AbstractSparseArray{T, N}},
    Adjoint{T, <:AbstractSparseArray},
    Transpose{T, <:AbstractSparseArray},
}
const AnyAbstractSparseArray{T, N} = Union{
    AbstractSparseArray{T, N}, WrappedAbstractSparseArray{T, N},
}
const AnyAbstractSparseVector{T} = AnyAbstractSparseArray{T, 1}
const AnyAbstractSparseMatrix{T} = AnyAbstractSparseArray{T, 2}
const AnyAbstractSparseVecOrMat{T} = Union{
    AnyAbstractSparseVector{T}, AnyAbstractSparseMatrix{T},
}

Base.convert(T::Type{<:AbstractSparseArray}, a::AbstractArray) = a isa T ? a : T(a)

using FunctionImplementations: FunctionImplementations
function FunctionImplementations.ImplementationStyle(::Type{<:AnyAbstractSparseArray})
    return SparseArrayImplementationStyle()
end

function Base.copy(a::AnyAbstractSparseArray)
    return copyto!(similar(a), a)
end

function Base.similar(a::AnyAbstractSparseArray, unstored::Unstored)
    return SparseArrayDOK(undef, unstored)
end
function Base.similar(a::AnyAbstractSparseArray)
    return similar(a, Unstored(unstored(a)))
end
function Base.similar(a::AnyAbstractSparseArray, T::Type)
    return similar(a, Unstored(unstoredsimilar(unstored(a), T)))
end
function Base.similar(a::AnyAbstractSparseArray, ax::Tuple)
    return similar(a, Unstored(unstoredsimilar(unstored(a), ax)))
end

function similar_sparsearray(a::AbstractArray, T::Type, ax::Tuple)
    return similar(a, Unstored(unstoredsimilar(unstored(a), T, ax)))
end
function Base.similar(a::AnyAbstractSparseArray, T::Type, ax::Tuple{Vararg{Int}})
    return similar_sparsearray(a, T, ax)
end
function Base.similar(
        a::AnyAbstractSparseArray, T::Type, ax::Tuple{Integer, Vararg{Integer}}
    )
    return similar_sparsearray(a, T, ax)
end
function Base.similar(
        a::AnyAbstractSparseArray,
        T::Type,
        ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}}
    )
    return similar_sparsearray(a, T, ax)
end

using ArrayLayouts: ArrayLayouts
using LinearAlgebra: LinearAlgebra

Base.getindex(a::AnyAbstractSparseArray, I::Any...) = style(a)(getindex)(a, I...)
Base.getindex(a::AnyAbstractSparseArray, I::Int...) = style(a)(getindex)(a, I...)
Base.setindex!(a::AnyAbstractSparseArray, x, I::Any...) = style(a)(setindex!)(a, x, I...)
Base.setindex!(a::AnyAbstractSparseArray, x, I::Int...) = style(a)(setindex!)(a, x, I...)
Base.copy!(dst::AbstractArray, src::AnyAbstractSparseArray) = style(src)(copy!)(dst, src)
function Base.copyto!(dst::AbstractArray, src::AnyAbstractSparseArray)
    return style(src)(copyto!)(dst, src)
end
Base.map(f, as::AnyAbstractSparseArray...) = style(as...)(map)(f, as...)
function Base.map!(f, dst::AbstractArray, as::AnyAbstractSparseArray...)
    return style(as...)(map!)(f, dst, as...)
end
function Base.mapreduce(f, op, as::AnyAbstractSparseArray...; kwargs...)
    return style(as...)(mapreduce)(f, op, as...; kwargs...)
end
function Base.reduce(f, as::AnyAbstractSparseArray...; kwargs...)
    return style(as...)(reduce)(f, as...; kwargs...)
end
Base.all(f::Function, a::AnyAbstractSparseArray) = style(a)(all)(f, a)
Base.all(a::AnyAbstractSparseArray) = style(a)(all)(a)
Base.iszero(a::AnyAbstractSparseArray) = style(a)(iszero)(a)
Base.isreal(a::AnyAbstractSparseArray) = style(a)(isreal)(a)
Base.real(a::AnyAbstractSparseArray) = style(a)(real)(a)
Base.fill!(a::AnyAbstractSparseArray, x) = style(a)(fill!)(a, x)
FunctionImplementations.zero!(a::AnyAbstractSparseArray) = style(a)(zero!)(a)
Base.zero(a::AnyAbstractSparseArray) = style(a)(zero)(a)
function Base.permutedims!(dst, a::AnyAbstractSparseArray, perm)
    return style(a)(permutedims!)(dst, a, perm)
end
function LinearAlgebra.mul!(
        dst::AbstractMatrix, a1::AnyAbstractSparseArray, a2::AnyAbstractSparseArray,
        α::Number, β::Number
    )
    return style(a1, a2)(mul!)(dst, a1, a2, α, β)
end

function Base.Broadcast.BroadcastStyle(type::Type{<:AnyAbstractSparseArray})
    return SparseArrayStyle{ndims(type)}()
end

using ArrayLayouts: ArrayLayouts
ArrayLayouts.MemoryLayout(type::Type{<:AnyAbstractSparseArray}) = SparseLayout()

using FunctionImplementations.Concatenate: concatenate
# We overload `Base._cat` instead of `Base.cat` since it
# is friendlier for invalidations/compile times, see:
# https://github.com/ITensor/SparseArraysBase.jl/issues/25
Base._cat(dims, a::AnyAbstractSparseArray...) = concatenate(dims, a...)

# TODO: Use `map(WeakPreserving(f), a)` instead.
# Currently that has trouble with type unstable maps, since
# the element type becomes abstract and therefore the zero/unstored
# values are not well defined.
function map_stored(f, a::AnyAbstractSparseArray)
    iszero(storedlength(a)) && return a
    kvs = storedpairs(a)
    # `collect` to convert to `Vector`, since otherwise
    # if it stays as `Dictionary` we might hit issues like
    # https://github.com/andyferris/Dictionaries.jl/issues/163.
    ks = collect(first.(kvs))
    vs = collect(last.(kvs))
    vs′ = map(f, vs)
    a′ = zero!(similar(a, eltype(vs′)))
    for (k, v′) in zip(ks, vs′)
        a′[k] = v′
    end
    return a′
end

using Adapt: adapt
function Base.print_array(io::IO, a::AnyAbstractSparseArray)
    # TODO: Use `map(WeakPreserving(adapt(Array)), a)` instead.
    # Currently that has trouble with type unstable maps, since
    # the element type becomes abstract and therefore the zero/unstored
    # values are not well defined.
    a′ = map_stored(adapt(Array), a)
    return @invoke Base.print_array(io::typeof(io), a′::AbstractArray{<:Any, ndims(a)})
end
function Base.replace_in_print_matrix(
        a::AnyAbstractSparseVecOrMat, i::Integer, j::Integer, s::AbstractString
    )
    return isstored(a, i, j) ? s : Base.replace_with_centered_mark(s)
end

# Special-purpose constructors
# ----------------------------

"""
    sparse(storage::Union{AbstractDict,AbstractDictionary}, dims...[; getunstored])

Construct an `N`-dimensional [`SparseArrayDOK`](@ref) containing elements of type `T`. Both
`T` and `N` can either be supplied explicitly or be determined by the `storage` and the
length or number of `dims`. If `dims` aren't specified, the size will be determined automatically
from the input indices.

This constructor does not take ownership of the supplied storage, and will result in an
independent container.
"""
sparse(::Union{AbstractDict, AbstractDictionary}, dims...)

const AbstractDictOrDictionary = Union{AbstractDict, AbstractDictionary}
# checked constructor from data: use `setindex!` to validate/convert input
function sparse(storage::AbstractDictOrDictionary, unstored::AbstractArray)
    A = SparseArrayDOK(undef, Unstored(unstored))
    for (i, v) in pairs(storage)
        A[i] = v
    end
    return A
end
function sparse(storage::AbstractDictOrDictionary, ax::Tuple)
    return sparse(storage, Zeros{valtype(storage)}(ax))
end
function sparse(storage::AbstractDictOrDictionary, dims::Int...)
    return sparse(storage, dims)
end
# Determine the size automatically.
function sparse(storage::AbstractDictOrDictionary)
    dims = ntuple(Returns(0), length(keytype(storage)))
    for I in keys(storage)
        dims = map(max, dims, Tuple(I))
    end
    return sparse(storage, dims)
end

using Random: Random, AbstractRNG, default_rng

@doc """
    sparsezeros([T::Type], dims[; getunstored]) -> A::SparseArrayDOK{T}

Create an empty size `dims` sparse array.
The optional `T` argument specifies the element type, which defaults to `Float64`.
""" sparsezeros

function sparsezeros(::Type{T}, unstored::AbstractArray{<:Any, N}) where {T, N}
    return SparseArrayDOK{T, N}(undef, Unstored(unstored))
end
function sparsezeros(unstored::AbstractArray{T, N}) where {T, N}
    return SparseArrayDOK{T, N}(undef, Unstored(unstored))
end
function sparsezeros(::Type{T}, dims::Dims) where {T}
    return sparsezeros(T, Zeros{T}(dims))
end
sparsezeros(::Type{T}, dims::Int...) where {T} = sparsezeros(T, dims)
sparsezeros(dims::Dims) = sparsezeros(Float64, dims)
sparsezeros(dims::Int...) = sparsezeros(Float64, dims)

@doc """
    sparserand([rng], [T::Type], dims; density::Real=0.5, randfun::Function=rand) -> A::SparseArrayDOK{T}

Create a random size `dims` sparse array in which the probability of any element being stored is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `T` argument specifies the element type, which defaults to `Float64`.
The optional `randfun` argument can be used to control the type of random elements, and should support
the signature `randfun(rng, T, N)` to generate `N` entries of type `T`.


See also [`sparserand!`](@ref).
""" sparserand

function sparserand(::Type{T}, dims::Dims; kwargs...) where {T}
    return sparserand(default_rng(), T, dims; kwargs...)
end
function sparserand(::Type{T}, dims::Int...; kwargs...) where {T}
    return sparserand(T, dims; kwargs...)
end
sparserand(dims::Dims; kwargs...) = sparserand(default_rng(), Float64, dims; kwargs...)
sparserand(dims::Int...; kwargs...) = sparserand(dims; kwargs...)
function sparserand(rng::AbstractRNG, dims::Dims; kwargs...)
    return sparserand(rng, Float64, dims; kwargs...)
end
function sparserand(rng::AbstractRNG, dims::Int...; kwargs...)
    return sparserand(rng, dims; kwargs...)
end
function sparserand(rng::AbstractRNG, ::Type{T}, dims::Dims; kwargs...) where {T}
    A = SparseArrayDOK{T}(undef, dims)
    sparserand!(rng, A; kwargs...)
    return A
end
function sparserand(rng::AbstractRNG, ::Type{T}, dims::Int...; kwargs...) where {T}
    return sparserand(rng, T, dims; kwargs...)
end

@doc """
    sparserand!([rng], A::AbstractArray; density::Real=0.5, randfun::Function=rand) -> A

Overwrite part of an array with random entries, where the probability of overwriting is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `randfun` argument can be used to control the type of random elements, and should support
the signature `randfun(rng, T, N)` to generate `N` entries of type `T`.

See also [`sparserand`](@ref).
""" sparserand!

sparserand!(A::AbstractArray; kwargs...) = sparserand!(default_rng(), A; kwargs...)
function sparserand!(
        rng::AbstractRNG, A::AbstractArray; density::Real = 0.5, randfun::Function = Random.rand
    )
    ArrayLayouts.zero!(A)
    rand_inds = Random.randsubseq(rng, eachindex(A), density)
    rand_entries = randfun(rng, eltype(A), length(rand_inds))
    return @inbounds for (I, v) in zip(rand_inds, rand_entries)
        A[I] = v
    end
end

using ArrayLayouts: ArrayLayouts, MemoryLayout
using LinearAlgebra: LinearAlgebra, Adjoint
function ArrayLayouts.MemoryLayout(
        ::Type{Transpose{T, P}}
    ) where {T, P <: AbstractSparseMatrix}
    return MemoryLayout(P)
end
function ArrayLayouts.MemoryLayout(
        ::Type{Adjoint{T, P}}
    ) where {T, P <: AbstractSparseMatrix}
    return MemoryLayout(P)
end
function LinearAlgebra.mul!(
        dest::AbstractMatrix,
        A::Adjoint{<:Any, <:AbstractSparseMatrix}, B::AbstractSparseMatrix,
        α::Number, β::Number
    )
    return ArrayLayouts.mul!(dest, A, B, α, β)
end
function LinearAlgebra.mul!(
        dest::AbstractMatrix,
        A::AbstractSparseMatrix,
        B::Adjoint{<:Any, <:AbstractSparseMatrix},
        α::Number,
        β::Number
    )
    return ArrayLayouts.mul!(dest, A, B, α, β)
end
function LinearAlgebra.mul!(
        dest::AbstractMatrix,
        A::Adjoint{<:Any, <:AbstractSparseMatrix},
        B::Adjoint{<:Any, <:AbstractSparseMatrix},
        α::Number,
        β::Number
    )
    return ArrayLayouts.mul!(dest, A, B, α, β)
end
