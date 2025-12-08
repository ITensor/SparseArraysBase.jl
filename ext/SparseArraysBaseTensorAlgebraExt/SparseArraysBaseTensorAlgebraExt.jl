module SparseArraysBaseTensorAlgebraExt

using SparseArrays: SparseMatrixCSC
using SparseArraysBase: AnyAbstractSparseArray, AnyAbstractSparseMatrix, SparseArrayDOK
using TensorAlgebra: TensorAlgebra, BlockedTrivialPermutation, BlockedTuple, FusionStyle,
    ReshapeFusion, matricize, unmatricize

struct SparseArrayFusion <: FusionStyle end
TensorAlgebra.FusionStyle(::Type{<:AnyAbstractSparseArray}) = SparseArrayFusion()

function TensorAlgebra.matricize(
        style::SparseArrayFusion, a::AbstractArray, length_codomain::Val
    )
    m = matricize(ReshapeFusion(), a, length_codomain)
    return convert(SparseMatrixCSC, m)
end
function TensorAlgebra.unmatricize(
        style::SparseArrayFusion,
        m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    a = unmatricize(ReshapeFusion(), m, axes_codomain, axes_domain)
    # TODO: Use `similar_type(m)` instead of hardcoding to `SparseArrayDOK`?
    return convert(SparseArrayDOK, a)
end

end
