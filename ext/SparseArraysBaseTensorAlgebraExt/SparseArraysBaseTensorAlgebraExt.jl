module SparseArraysBaseTensorAlgebraExt

using SparseArrays: SparseMatrixCSC
using SparseArraysBase: AnyAbstractSparseArray, AnyAbstractSparseMatrix, SparseArrayDOK
using TensorAlgebra:
    TensorAlgebra, FusionStyle, ReshapeFusion, bipermutedimsopadd!, matricize, unmatricize

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
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    a = unmatricize(ReshapeFusion(), m, axes_codomain, axes_domain)
    # TODO: Use `similar_type(m)` instead of hardcoding to `SparseArrayDOK`?
    return convert(SparseArrayDOK, a)
end

# A sparse array can't be wrapped in a `StridedView`, so the generic
# `bipermutedimsopadd!` doesn't apply. Accumulate over a lazily permuted source via
# broadcasting, which dispatches to the sparse broadcast path. `_opadd!` mirrors the
# accumulation in TensorAlgebra's generic method.
function TensorAlgebra.bipermutedimsopadd!(
        dest::AnyAbstractSparseArray, op, src::AbstractArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    perm = (perm_codomain..., perm_domain...)
    _opadd!(dest, op, PermutedDimsArray(src, perm), α, β)
    return dest
end

function _opadd!(dest::AbstractArray, op, src::AbstractArray, α, β)
    if op === identity
        if iszero(β)
            dest .= α .* src
        else
            dest .= β .* dest .+ α .* src
        end
    else
        if iszero(β)
            dest .= α .* op.(src)
        else
            dest .= β .* dest .+ α .* op.(src)
        end
    end
    return dest
end

end
