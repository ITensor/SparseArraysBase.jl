using LinearAlgebra: Adjoint
storedvalues(a::Adjoint) = storedvalues(parent(a))
function isstored(a::Adjoint, i::Int, j::Int)
  return isstored(parent(a), j, i)
end
function eachstoredindex(a::Adjoint)
  # TODO: Make lazy with `Iterators.map`.
  return map(CartesianIndex ∘ reverse ∘ Tuple, collect(eachstoredindex(parent(a))))
end
function getstoredindex(a::Adjoint, i::Int, j::Int)
  return getstoredindex(parent(a), j, i)'
end
function getunstoredindex(a::Adjoint, i::Int, j::Int)
  return getunstoredindex(parent(a), j, i)'
end
function setstoredindex!(a::Adjoint, value, i::Int, j::Int)
  setstoredindex!(parent(a), value', j, i)
  return a
end
function setunstoredindex!(a::Adjoint, value, i::Int, j::Int)
  setunstoredindex!(parent(a), value', j, i)
  return a
end

using LinearAlgebra: Transpose
storedvalues(a::Transpose) = storedvalues(parent(a))
function isstored(a::Transpose, i::Int, j::Int)
  return isstored(parent(a), j, i)
end
function eachstoredindex(a::Transpose)
  # TODO: Make lazy with `Iterators.map`.
  return map(CartesianIndex ∘ reverse ∘ Tuple, collect(eachstoredindex(parent(a))))
end
function getstoredindex(a::Transpose, i::Int, j::Int)
  return transpose(getstoredindex(parent(a), j, i))
end
function getunstoredindex(a::Transpose, i::Int, j::Int)
  return transpose(getunstoredindex(parent(a), j, i))
end
function setstoredindex!(a::Transpose, value, i::Int, j::Int)
  setstoredindex!(parent(a), transpose(value), j, i)
  return a
end
function setunstoredindex!(a::Transpose, value, i::Int, j::Int)
  setunstoredindex!(parent(a), transpose(value), j, i)
  return a
end
