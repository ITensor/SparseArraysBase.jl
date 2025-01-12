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

# This type alias is a temporary workaround since `@derive`
# doesn't parse the `@MIME_str` macro properly at the moment.
const MIMEtextplain = MIME"text/plain"
@derive (T=AnyAbstractSparseArray,) begin
  Base.show(::IO, ::MIMEtextplain, ::T)
end

# Wraps a sparse array but replaces the unstored values.
# This is used in printing in order to customize printing
# of zero/unstored values.
struct ReplacedUnstoredSparseArray{T,N,F,Parent<:AbstractArray{T,N}} <:
       AbstractSparseArray{T,N}
  parent::Parent
  getunstoredindex::F
end
Base.parent(a::ReplacedUnstoredSparseArray) = a.parent
Base.size(a::ReplacedUnstoredSparseArray) = size(parent(a))
function isstored(a::ReplacedUnstoredSparseArray, I::Int...)
  return isstored(parent(a), I...)
end
function getstoredindex(a::ReplacedUnstoredSparseArray, I::Int...)
  return getstoredindex(parent(a), I...)
end
function getunstoredindex(a::ReplacedUnstoredSparseArray, I::Int...)
  return a.getunstoredindex(a, I...)
end
@derive ReplacedUnstoredSparseArray AbstractArrayOps
