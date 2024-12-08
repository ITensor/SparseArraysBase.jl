abstract type AbstractSparseArray{T,N} <: AbstractArray{T,N} end

using Derive: @array_aliases
# Define AbstractSparseVector, AnyAbstractSparseArray, etc.
@array_aliases AbstractSparseArray

using Derive: Derive
function Derive.interface(::Type{<:AbstractSparseArray})
  return SparseArrayInterface()
end

using Derive: @derive

# TODO: These need to be loaded since `AbstractArrayOps`
# includes overloads of functions from these modules.
# Ideally that wouldn't be needed and can be circumvented
# with `GlobalRef`.
using ArrayLayouts: ArrayLayouts
using LinearAlgebra: LinearAlgebra

# Derive `Base.getindex`, `Base.setindex!`, etc.
# TODO: Define `AbstractMatrixOps` and overload for
# `AnyAbstractSparseMatrix` and `AnyAbstractSparseVector`,
# which is where matrix multiplication and factorizations
# shoudl go.
@derive AnyAbstractSparseArray AbstractArrayOps
