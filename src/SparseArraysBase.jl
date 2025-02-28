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
  storedlength,
  storedpairs,
  storedvalues,
  sparsezeros,
  sparserand,
  sparserand!

include("abstractsparsearrayinterface.jl")
include("sparsearrayinterface.jl")
include("wrappers.jl")
include("abstractsparsearray.jl")
include("sparsearraydok.jl")
include("oneelementarray.jl")

end
