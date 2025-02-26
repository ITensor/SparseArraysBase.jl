module SparseArraysBase

export SparseArrayDOK,
  SparseMatrixDOK,
  SparseVectorDOK,
  OneElementArray,
  OneElementMatrix,
  OneElementVector,
  eachstoredindex,
  isstored,
  oneelementarray,
  storedlength,
  storedpairs,
  storedvalues

using DerivableInterfaces

# sparsearrayinterface
include("abstractsparsearrayinterface.jl")
include("indexing.jl")
include("map.jl")
include("sparsearrayinterface.jl")

# types
include("wrappers.jl")
include("abstractsparsearray.jl")
include("sparsearraydok.jl")
include("oneelementarray.jl")

end
