using Derive: Derive

struct SparseArrayInterface <: AbstractSparseArrayInterface end

# Convenient shorthand to refer to the sparse interface.
# Can turn a function into a sparse function with the syntax `sparse(f)`,
# i.e. `sparse(map)(x -> 2x, randn(2, 2))` while use the sparse
# version of `map`.
# const sparse = SparseArrayInterface()

Derive.interface(::AbstractSparseArrayStyle) = SparseArrayInterface()
