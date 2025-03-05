parentvalue_to_value(a::AbstractArray, value) = value
value_to_parentvalue(a::AbstractArray, value) = value
eachstoredparentindex(a::AbstractArray) = eachstoredindex(parent(a))
storedparentvalues(a::AbstractArray) = storedvalues(parent(a))

function parentindex_to_index(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  return throw(MethodError(parentindex_to_index, Tuple{typeof(a),typeof(I)}))
end
function parentindex_to_index(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return Tuple(parentindex_to_index(a, CartesianIndex(I)))
end
# Handle linear indexing.
function parentindex_to_index(a::AbstractArray, I::Int)
  return parentindex_to_index(a, CartesianIndices(parent(a))[I])
end

function index_to_parentindex(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  return throw(MethodError(index_to_parentindex, Tuple{typeof(a),typeof(I)}))
end
function index_to_parentindex(a::AbstractArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return Tuple(index_to_parentindex(a, CartesianIndex(I)))
end
# Handle linear indexing.
function index_to_parentindex(a::AbstractArray, I::Int)
  return index_to_parentindex(a, CartesianIndices(a)[I])
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

perm(::PermutedDimsArray{<:Any,<:Any,p}) where {p} = p
iperm(::PermutedDimsArray{<:Any,<:Any,<:Any,ip}) where {ip} = ip
function index_to_parentindex(a::PermutedDimsArray{<:Any,N}, I::CartesianIndex{N}) where {N}
  return CartesianIndex(genperm(I, iperm(a)))
end
function parentindex_to_index(a::PermutedDimsArray{<:Any,N}, I::CartesianIndex{N}) where {N}
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
    end,
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

@interface ::AbstractArrayInterface function isstored(a::SubArray, I::Int...)
  return isstored(parent(a), index_to_parentindex(a, I...)...)
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

# TODO: Turn these into `AbstractWrappedSparseArrayInterface` functions?
for type in (:Adjoint, :PermutedDimsArray, :ReshapedArray, :SubArray, :Transpose)
  @eval begin
    @interface ::AbstractSparseArrayInterface storedvalues(a::$type) = storedparentvalues(a)
    @interface ::AbstractSparseArrayInterface function isstored(a::$type, I::Int...)
      return isstored(parent(a), index_to_parentindex(a, I...)...)
    end
    @interface ::AbstractSparseArrayInterface function eachstoredindex(a::$type)
      # TODO: Make lazy with `Iterators.map`.
      return map(collect(eachstoredparentindex(a))) do I
        return parentindex_to_index(a, I)
      end
    end
    @interface ::AbstractSparseArrayInterface function getstoredindex(a::$type, I::Int...)
      return parentvalue_to_value(
        a, getstoredindex(parent(a), index_to_parentindex(a, I...)...)
      )
    end
    @interface ::AbstractSparseArrayInterface function getunstoredindex(a::$type, I::Int...)
      return parentvalue_to_value(
        a, getunstoredindex(parent(a), index_to_parentindex(a, I...)...)
      )
    end
    @interface ::AbstractSparseArrayInterface function setstoredindex!(
      a::$type, value, I::Int...
    )
      setstoredindex!(
        parent(a), value_to_parentvalue(a, value), index_to_parentindex(a, I...)...
      )
      return a
    end
    @interface ::AbstractSparseArrayInterface function setunstoredindex!(
      a::$type, value, I::Int...
    )
      setunstoredindex!(
        parent(a), value_to_parentvalue(a, value), index_to_parentindex(a, I...)...
      )
      return a
    end
  end
end

using LinearAlgebra: LinearAlgebra, Diagonal
@interface ::AbstractArrayInterface storedvalues(D::Diagonal) = LinearAlgebra.diag(D)

# compat with LTS:
@static if VERSION ≥ v"1.11"
  _diagind = LinearAlgebra.diagind
else
  function _diagind(x::Diagonal, ::IndexCartesian)
    return view(CartesianIndices(x), LinearAlgebra.diagind(x))
  end
end
@interface ::AbstractArrayInterface eachstoredindex(D::Diagonal) =
  _diagind(D, IndexCartesian())

@interface ::AbstractArrayInterface isstored(D::Diagonal, i::Int, j::Int) =
  i == j && Base.checkbounds(Bool, D, i, j)
@interface ::AbstractArrayInterface function getstoredindex(D::Diagonal, i::Int, j::Int)
  return D.diag[i]
end
@interface ::AbstractArrayInterface function getunstoredindex(D::Diagonal, i::Int, j::Int)
  return zero(eltype(D))
end
@interface ::AbstractArrayInterface function setstoredindex!(D::Diagonal, v, i::Int, j::Int)
  D.diag[i] = v
  return D
end
