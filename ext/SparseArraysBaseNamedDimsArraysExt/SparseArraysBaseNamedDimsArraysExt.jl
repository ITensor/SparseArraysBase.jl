module SparseArraysBaseNamedDimsArraysExt

using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedUnitRange, dename, inds,
    nameddims, nameddimsof
using SparseArraysBase: SparseArraysBase, dense, oneelement

function SparseArraysBase.dense(a::AbstractNamedDimsArray)
    # TODO: Use `NamedDimsArrays.nameddimsof(a, dense(unname(a)))` once that is defined,
    # see: https://github.com/ITensor/NamedDimsArrays.jl/issues/138
    return nameddimsof(a, dense(dename(a)))
end

function SparseArraysBase.oneelement(
        value, index::NTuple{N, Int}, ax::NTuple{N, AbstractNamedUnitRange}
    ) where {N}
    return nameddims(oneelement(value, index, only.(axes.(dename.(ax)))), ax)
end

end
