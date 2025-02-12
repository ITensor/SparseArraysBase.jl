abstract type AbstractSparseArray{T,N} <: AbstractArray{T,N} end

using DerivableInterfaces: @array_aliases
# Define AbstractSparseVector, AnyAbstractSparseArray, etc.
@array_aliases AbstractSparseArray

using DerivableInterfaces: DerivableInterfaces
function DerivableInterfaces.interface(::Type{<:AbstractSparseArray})
  return SparseArrayInterface()
end

using DerivableInterfaces: @derive

# TODO: These need to be loaded since `AbstractArrayOps`
# includes overloads of functions from these modules.
# Ideally that wouldn't be needed and can be circumvented
# with `GlobalRef`.
using ArrayLayouts: ArrayLayouts
using LinearAlgebra: LinearAlgebra

# DerivableInterfaces `Base.getindex`, `Base.setindex!`, etc.
# TODO: Define `AbstractMatrixOps` and overload for
# `AnyAbstractSparseMatrix` and `AnyAbstractSparseVector`,
# which is where matrix multiplication and factorizations
# should go.
@derive AnyAbstractSparseArray AbstractArrayOps

function Base.replace_in_print_matrix(
  A::AnyAbstractSparseArray{<:Any,2}, i::Integer, j::Integer, s::AbstractString
)
  return isstored(A, CartesianIndex(i, j)) ? s : Base.replace_with_centered_mark(s)
end
