module SparseArraysBaseNamedDimsArraysExt

using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedUnitRange,
    constructorof_nameddims, dename, inds, nameddims
using SparseArraysBase: SparseArraysBase, dense, oneelement

function SparseArraysBase.dense(a::AbstractNamedDimsArray)
    # TODO: Rewrite as `set_unnamed(a, dense(unname(a)))`.
    # See also on DimensionalData.jl's interface:
    # https://github.com/rafaqz/DimensionalData.jl/blob/v0.29.25/src/utils.jl#L93-L97
    return constructorof_nameddims(typeof(a))(dense(dename(a)), inds(a))
end

function SparseArraysBase.oneelement(
        value, index::NTuple{N, Int}, ax::NTuple{N, AbstractNamedUnitRange}
    ) where {N}
    return nameddims(oneelement(value, index, only.(axes.(dename.(ax)))), ax)
end

end
