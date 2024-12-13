function cartesianindex_reverse(I::CartesianIndex)
  return CartesianIndex(reverse(Tuple(I)))
end
tuple_oneto(n) = ntuple(identity, n)

# TODO: Use `Base.PermutedDimsArrays.genperm` or
# https://github.com/jipolanco/StaticPermutations.jl?
genperm(v, perm) = map(j -> v[j], perm)

## TODO: Use this and something similar for `Dictionary` to make a faster
## implementation of `storedvalues(::SubArray)`.
## function valuesview(d::Dict, keys)
##   return @view d.vals[[Base.ht_keyindex(d, key) for key in keys]]
## end

function eachstoredparentindex(a::SubArray)
  return filter(eachstoredindex(parent(a))) do I
    return all(d -> I[d] ∈ parentindices(a)[d], 1:ndims(parent(a)))
  end
end
function storedvalues(a::SubArray)
  # We use `StoredValues` rather than `@view`/`SubArray` so that
  # it gets interpreted as a dense array.
  return StoredValues(parent(a), collect(eachstoredparentindex(a)))
end
function isstored(a::SubArray, I::Int...)
  return isstored(parent(a), Base.reindex(parentindices(a), I)...)
end
function getstoredindex(a::SubArray, I::Int...)
  return getstoredindex(parent(a), Base.reindex(parentindices(a), I)...)
end
function getunstoredindex(a::SubArray, I::Int...)
  return getunstoredindex(parent(a), Base.reindex(parentindices(a), I)...)
end
function eachstoredindex(a::SubArray)
  nonscalardims = filter(tuple_oneto(ndims(parent(a)))) do d
    return !(parentindices(a)[d] isa Real)
  end
  return collect((
    CartesianIndex(
      map(nonscalardims) do d
        return findfirst(==(I[d]), parentindices(a)[d])
      end,
    ) for I in eachstoredparentindex(a)
  ))
end

perm(::PermutedDimsArray{<:Any,<:Any,p}) where {p} = p
iperm(::PermutedDimsArray{<:Any,<:Any,<:Any,ip}) where {ip} = ip

storedvalues(a::PermutedDimsArray) = storedvalues(parent(a))
function isstored(a::PermutedDimsArray, I::Int...)
  return isstored(parent(a), genperm(I, iperm(a))...)
end
function getstoredindex(a::PermutedDimsArray, I::Int...)
  return getstoredindex(parent(a), genperm(I, iperm(a))...)
end
function getunstoredindex(a::PermutedDimsArray, I::Int...)
  return getunstoredindex(parent(a), genperm(I, iperm(a))...)
end
function setstoredindex!(a::PermutedDimsArray, value, I::Int...)
  # TODO: Should this be `iperm(a)`?
  setstoredindex!(parent(a), value, genperm(I, perm(a))...)
  return a
end
function setunstoredindex!(a::PermutedDimsArray, value, I::Int...)
  # TODO: Should this be `iperm(a)`?
  setunstoredindex!(parent(a), value, genperm(I, perm(a))...)
  return a
end
function eachstoredindex(a::PermutedDimsArray)
  # TODO: Make lazy with `Iterators.map`.
  return map(collect(eachstoredindex(parent(a)))) do I
    return CartesianIndex(genperm(I, perm(a)))
  end
end

for (type, func) in ((:Adjoint, :adjoint), (:Transpose, :transpose))
  @eval begin
    using LinearAlgebra: $type
    storedvalues(a::$type) = storedvalues(parent(a))
    function isstored(a::$type, i::Int, j::Int)
      return isstored(parent(a), j, i)
    end
    function eachstoredindex(a::$type)
      # TODO: Make lazy with `Iterators.map`.
      return map(cartesianindex_reverse, collect(eachstoredindex(parent(a))))
    end
    function getstoredindex(a::$type, i::Int, j::Int)
      return $func(getstoredindex(parent(a), j, i))
    end
    function getunstoredindex(a::$type, i::Int, j::Int)
      return $func(getunstoredindex(parent(a), j, i))
    end
    function setstoredindex!(a::$type, value, i::Int, j::Int)
      setstoredindex!(parent(a), $func(value), j, i)
      return a
    end
    function setunstoredindex!(a::$type, value, i::Int, j::Int)
      setunstoredindex!(parent(a), $func(value), j, i)
      return a
    end
  end
end
