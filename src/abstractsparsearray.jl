abstract type AbstractSparseArray{T,N} <: AbstractArray{T,N} end

using Derive: @array_aliases
# Define AbstractSparseVector, AnyAbstractSparseArray, etc.
@array_aliases AbstractSparseArray

using Derive: Derive
function Derive.interface(::Type{<:AbstractSparseArray})
  return SparseArrayInterface()
end

using Derive: @derive
# Derive `Base.getindex`, `Base.setindex!`, etc.
@derive AnyAbstractSparseArray AbstractArrayOps

using LinearAlgebra: LinearAlgebra
@derive (T=AnyAbstractSparseVecOrMat,) begin
  LinearAlgebra.mul!(::AbstractMatrix, ::T, ::T, ::Number, ::Number)
end

using ArrayLayouts: ArrayLayouts
@derive (T=AnyAbstractSparseArray,) begin
  ArrayLayouts.MemoryLayout(::Type{<:T})
end
