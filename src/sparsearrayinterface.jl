using DerivableInterfaces: DerivableInterfaces

struct SparseArrayInterface{N} <: AbstractSparseArrayInterface{N} end
SparseArrayInterface() = SparseArrayInterface{Any}()
SparseArrayInterface(::Val{N}) where {N} = SparseArrayInterface{N}()
SparseArrayInterface{M}(::Val{N}) where {M,N} = SparseArrayInterface{N}()

# Fix ambiguity error.
function DerivableInterfaces.combine_interface_rule(
  ::SparseArrayInterface{N}, ::SparseArrayInterface{N}
) where {N}
  return SparseArrayInterface{N}()
end
function DerivableInterfaces.combine_interface_rule(
  ::SparseArrayInterface, ::SparseArrayInterface
)
  return SparseArrayInterface()
end
function DerivableInterfaces.combine_interface_rule(
  interface1::SparseArrayInterface, interface2::AbstractSparseArrayInterface
)
  return interface1
end
function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractSparseArrayInterface, interface2::SparseArrayInterface
)
  return interface2
end

# Convenient shorthand to refer to the sparse interface.
# Can turn a function into a sparse function with the syntax `sparse(f)`,
# i.e. `sparse(map)(x -> 2x, randn(2, 2))` while use the sparse
# version of `map`.
# const sparse = SparseArrayInterface()

DerivableInterfaces.interface(::Type{<:AbstractSparseArrayStyle}) = SparseArrayInterface()
