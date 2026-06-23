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

# `zero!` isn't defined in `Base`, but it is defined in `ArrayLayouts`
# and is useful for sparse array logic, since it can be used to empty
# the sparse array storage. SparseArraysBase owns its own `zero!` rather
# than relying on an external definition.
function zero! end

include("concatenate.jl")
include("abstractsparsearraystyle.jl")
include("sparsearraystyle.jl")
include("indexing.jl")
include("map.jl")
include("wrappers.jl")
include("abstractsparsearray.jl")
include("sparsearraydok.jl")
include("oneelementarray.jl")
include("sparsearrays.jl")

end
