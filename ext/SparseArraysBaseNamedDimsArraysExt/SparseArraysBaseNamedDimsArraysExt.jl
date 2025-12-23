module SparseArraysBaseNamedDimsArraysExt

using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedUnitRange,
    constructorof_nameddims, dename, inds, nameddims
using SparseArraysBase: SparseArraysBase, dense, oneelement

function SparseArraysBase.dense(a::AbstractNamedDimsArray)
    # TODO: Use `NamedDimsArrays.nameddimsof(a, dense(unname(a)))` once that is defined,
    # see: https://github.com/ITensor/NamedDimsArrays.jl/issues/138
    return constructorof_nameddims(typeof(a))(dense(dename(a)), inds(a))
end

function SparseArraysBase.oneelement(
        value, index::NTuple{N, Int}, ax::NTuple{N, AbstractNamedUnitRange}
    ) where {N}
    return nameddims(oneelement(value, index, only.(axes.(dename.(ax)))), ax)
end

end
