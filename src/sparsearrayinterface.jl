using Derive: Derive

struct SparseArrayInterface <: AbstractSparseArrayInterface end

# Fix ambiguity error.
function Derive.combine_interface_rule(::SparseArrayInterface, ::SparseArrayInterface)
  return SparseArrayInterface()
end
function Derive.combine_interface_rule(
  interface1::SparseArrayInterface, interface2::AbstractSparseArrayInterface
)
  return interface1
end
function Derive.combine_interface_rule(
  interface1::AbstractSparseArrayInterface, interface2::SparseArrayInterface
)
  return interface2
end

# Convenient shorthand to refer to the sparse interface.
# Can turn a function into a sparse function with the syntax `sparse(f)`,
# i.e. `sparse(map)(x -> 2x, randn(2, 2))` while use the sparse
# version of `map`.
# const sparse = SparseArrayInterface()

Derive.interface(::Type{<:AbstractSparseArrayStyle}) = SparseArrayInterface()
