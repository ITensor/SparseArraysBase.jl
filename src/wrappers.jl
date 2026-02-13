parentvalue_to_value(a::AbstractArray, value) = value
value_to_parentvalue(a::AbstractArray, value) = value
eachstoredparentindex(a::AbstractArray) = eachstoredindex(parent(a))
function eachstoredparentindex(style::IndexStyle, a::AbstractArray)
    return eachstoredindex(style, parent(a))
end
storedparentvalues(a::AbstractArray) = storedvalues(parent(a))

function parentindex_to_index(a::AbstractArray{<:Any, N}, I::CartesianIndex{N}) where {N}
    return throw(MethodError(parentindex_to_index, Tuple{typeof(a), typeof(I)}))
end
function parentindex_to_index(a::AbstractArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    return Tuple(parentindex_to_index(a, CartesianIndex(I)))
end
# Handle linear indexing.
function parentindex_to_index(a::AbstractArray, I::Int)
    return parentindex_to_index(a, CartesianIndices(parent(a))[I])
end

function index_to_parentindex(a::AbstractArray{<:Any, N}, I::CartesianIndex{N}) where {N}
    return throw(MethodError(index_to_parentindex, Tuple{typeof(a), typeof(I)}))
end
function index_to_parentindex(a::AbstractArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    return Tuple(index_to_parentindex(a, CartesianIndex(I)))
end
# Handle linear indexing.
function index_to_parentindex(a::AbstractArray, I::Int)
    return LinearIndices(parent(a))[index_to_parentindex(a, CartesianIndices(a)[I])]
end

function cartesianindex_reverse(I::CartesianIndex)
    return CartesianIndex(reverse(Tuple(I)))
end
tuple_oneto(n) = ntuple(identity, n)

# TODO: Use `Base.PermutedDimsArrays.genperm` or
# https://github.com/jipolanco/StaticPermutations.jl?
genperm(v, perm) = map(j -> v[j], perm)

using LinearAlgebra: Adjoint
function parentindex_to_index(a::Adjoint, I::CartesianIndex{2})
    return cartesianindex_reverse(I)
end
function index_to_parentindex(a::Adjoint, I::CartesianIndex{2})
    return cartesianindex_reverse(I)
end
function parentvalue_to_value(a::Adjoint, value)
    return adjoint(value)
end
function value_to_parentvalue(a::Adjoint, value)
    return adjoint(value)
end

perm(::PermutedDimsArray{<:Any, <:Any, p}) where {p} = p
iperm(::PermutedDimsArray{<:Any, <:Any, <:Any, ip}) where {ip} = ip
function index_to_parentindex(
        a::PermutedDimsArray{<:Any, N}, I::CartesianIndex{N}
    ) where {N}
    return CartesianIndex(genperm(I, iperm(a)))
end
function parentindex_to_index(
        a::PermutedDimsArray{<:Any, N}, I::CartesianIndex{N}
    ) where {N}
    return CartesianIndex(genperm(I, perm(a)))
end

using Base: ReshapedArray
# Don't constrain the number of dimensions of the array
# and index since the parent array can have a different
# number of dimensions than the `SubArray`.
function parentindex_to_index(a::ReshapedArray, I::CartesianIndex)
    return CartesianIndices(size(a))[LinearIndices(parent(a))[I]]
end
# Don't constrain the number of dimensions of the array
# and index since the parent array can have a different
# number of dimensions than the `SubArray`.
function index_to_parentindex(a::ReshapedArray, I::CartesianIndex)
    return CartesianIndices(parent(a))[LinearIndices(size(a))[I]]
end

function eachstoredparentindex(a::SubArray)
    return filter(eachstoredindex(parent(a))) do I
        return all(d -> I[d] ∈ parentindices(a)[d], 1:ndims(parent(a)))
    end
end
function eachstoredparentindex(style::IndexStyle, a::SubArray)
    return filter(eachstoredindex(style, parent(a))) do I
        return all(d -> I[d] ∈ parentindices(a)[d], 1:ndims(parent(a)))
    end
end
# Don't constrain the number of dimensions of the array
# and index since the parent array can have a different
# number of dimensions than the `SubArray`.
function index_to_parentindex(a::SubArray, I::CartesianIndex)
    return CartesianIndex(Base.reindex(parentindices(a), Tuple(I)))
end
# Don't constrain the number of dimensions of the array
# and index since the parent array can have a different
# number of dimensions than the `SubArray`.
function parentindex_to_index(a::SubArray, I::CartesianIndex)
    nonscalardims = filter(tuple_oneto(ndims(parent(a)))) do d
        return !(parentindices(a)[d] isa Real)
    end
    return CartesianIndex(
        map(nonscalardims) do d
            return findfirst(==(I[d]), parentindices(a)[d])
        end
    )
end
## TODO: Use this and something similar for `Dictionary` to make a faster
## implementation of `storedvalues(::SubArray)`.
## function valuesview(d::Dict, keys)
##   return @view d.vals[[Base.ht_keyindex(d, key) for key in keys]]
## end
function storedparentvalues(a::SubArray)
    # We use `StoredValues` rather than `@view`/`SubArray` so that
    # it gets interpreted as a dense array.
    return StoredValues(parent(a), collect(eachstoredparentindex(a)))
end

using LinearAlgebra: Transpose
function parentindex_to_index(a::Transpose, I::CartesianIndex{2})
    return cartesianindex_reverse(I)
end
function index_to_parentindex(a::Transpose, I::CartesianIndex{2})
    return cartesianindex_reverse(I)
end
function parentvalue_to_value(a::Transpose, value)
    return transpose(value)
end
function value_to_parentvalue(a::Transpose, value)
    return transpose(value)
end

function isstored_wrapped(a::AbstractArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    return isstored(parent(a), index_to_parentindex(a, I...)...)
end

function isstored(a::Adjoint, I::Vararg{Int, 2})
    return isstored_wrapped(a, I...)
end
function isstored(a::PermutedDimsArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    return isstored_wrapped(a, I...)
end
function isstored(a::ReshapedArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    return isstored_wrapped(a, I...)
end
function isstored(a::SubArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    return isstored_wrapped(a, I...)
end
function isstored(a::Transpose, I::Vararg{Int, 2})
    return isstored_wrapped(a, I...)
end

# TODO: Turn these into `AbstractWrappedSparseArrayStyle` functions?
for type in (:Adjoint, :PermutedDimsArray, :ReshapedArray, :SubArray, :Transpose)
    @eval begin
        storedvalues_sparse(a::$type) = storedparentvalues(a)
        function eachstoredindex_sparse(a::$type)
            return map(Base.Fix1(parentindex_to_index, a), eachstoredparentindex(a))
        end
        function eachstoredindex_sparse(
                style::IndexStyle, a::$type
            )
            # TODO: Make lazy with `Iterators.map`.
            return map(Base.Fix1(parentindex_to_index, a), eachstoredparentindex(style, a))
        end
        function getstoredindex_sparse(a::$type, I::Int...)
            return parentvalue_to_value(
                a, getstoredindex(parent(a), index_to_parentindex(a, I...)...)
            )
        end
        function getunstoredindex_sparse(a::$type, I::Int...)
            return parentvalue_to_value(
                a, getunstoredindex(parent(a), index_to_parentindex(a, I...)...)
            )
        end
        function setstoredindex!_sparse(
                a::$type, value, I::Int...
            )
            setstoredindex!(
                parent(a), value_to_parentvalue(a, value), index_to_parentindex(a, I...)...
            )
            return a
        end
        function setunstoredindex!_sparse(
                a::$type, value, I::Int...
            )
            setunstoredindex!(
                parent(a), value_to_parentvalue(a, value), index_to_parentindex(a, I...)...
            )
            return a
        end
    end
end

using FunctionImplementations: ImplementationStyle
using LinearAlgebra: LinearAlgebra, Diagonal
const diag_style = ImplementationStyle(Diagonal)
const storedvalues_diag = diag_style(storedvalues)
storedvalues_diag(D::AbstractMatrix) = LinearAlgebra.diag(D)

# compat with LTS:
@static if VERSION ≥ v"1.11"
    _diagind = LinearAlgebra.diagind
else
    function _diagind(x::AbstractMatrix, ::IndexCartesian)
        return view(CartesianIndices(x), LinearAlgebra.diagind(x))
    end
end
const eachstoredindex_diag = diag_style(eachstoredindex)
eachstoredindex_diag(D::AbstractMatrix) = _diagind(D, IndexCartesian())

const isstored_diag = diag_style(isstored)
function isstored_diag(D::AbstractMatrix, i::Int, j::Int)
    return i == j && checkbounds(Bool, D, i, j)
end
const getstoredindex_diag = diag_style(getstoredindex)
getstoredindex_diag(D::AbstractMatrix, i::Int, j::Int) = D.diag[i]
const getunstoredindex_diag = diag_style(getunstoredindex)
function getunstoredindex_diag(D::AbstractMatrix, i::Int, j::Int)
    return zero(eltype(D))
end
const setstoredindex!_diag = diag_style(setstoredindex!)
function setstoredindex!_diag(D::AbstractMatrix, v, i::Int, j::Int)
    D.diag[i] = v
    return D
end
