module SparseArraysBase

export SparseArrayDOK,
    SparseMatrixDOK,
    SparseVectorDOK,
    OneElementArray,
    OneElementMatrix,
    OneElementVector,
    eachstoredindex,
    isstored,
    oneelement,
    sparse,
    sparserand,
    sparserand!,
    sparsezeros,
    storedlength,
    storedpairs,
    storedvalues

# `zero!` is owned by TensorAlgebra; SparseArraysBase extends it so its sparse
# types can empty their storage in place rather than filling with zeros.
using TensorAlgebra: TensorAlgebra, zero!

include("abstractsparsearraystyle.jl")
include("sparsearraystyle.jl")
include("indexing.jl")
include("map.jl")
include("wrappers.jl")
include("abstractsparsearray.jl")
include("sparsearraydok.jl")
include("oneelementarray.jl")
include("sparsearrays.jl")
include("tensoralgebra.jl")

end
