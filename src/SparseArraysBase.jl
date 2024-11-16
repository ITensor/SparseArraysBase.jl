module SparseArraysBase
include("sparsearrayinterface/arraylayouts.jl")
include("sparsearrayinterface/densearray.jl")
include("sparsearrayinterface/vectorinterface.jl")
include("sparsearrayinterface/interface.jl")
include("sparsearrayinterface/interface_optional.jl")
include("sparsearrayinterface/indexing.jl")
include("sparsearrayinterface/base.jl")
include("sparsearrayinterface/map.jl")
include("sparsearrayinterface/copyto.jl")
include("sparsearrayinterface/broadcast.jl")
include("sparsearrayinterface/conversion.jl")
include("sparsearrayinterface/wrappers.jl")
include("sparsearrayinterface/zero.jl")
include("sparsearrayinterface/cat.jl")
include("sparsearrayinterface/SparseArraysBaseLinearAlgebraExt.jl")
include("abstractsparsearray/abstractsparsearray.jl")
include("abstractsparsearray/abstractsparsematrix.jl")
include("abstractsparsearray/abstractsparsevector.jl")
include("abstractsparsearray/wrappedabstractsparsearray.jl")
include("abstractsparsearray/arraylayouts.jl")
include("abstractsparsearray/sparsearrayinterface.jl")
include("abstractsparsearray/base.jl")
include("abstractsparsearray/broadcast.jl")
include("abstractsparsearray/map.jl")
include("abstractsparsearray/baseinterface.jl")
include("abstractsparsearray/convert.jl")
include("abstractsparsearray/cat.jl")
include("abstractsparsearray/SparseArraysBaseSparseArraysExt.jl")
include("abstractsparsearray/SparseArraysBaseLinearAlgebraExt.jl")
end
