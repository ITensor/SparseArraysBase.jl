using SparseArrays: SparseArrays, AbstractSparseMatrixCSC, SparseMatrixCSC, findnz

function eachstoredindex(m::AbstractSparseMatrixCSC)
      I, J, V = findnz(m)
    # TODO: This loses the compile time element type, is there a better lazy way?
    return Iterators.map(CartesianIndex, zip(I, J))
end
function eachstoredindex(a::Base.ReshapedArray{<:Any, <:Any, <:AbstractSparseMatrixCSC})
    return eachstoredindex_sparse(a)
end

function SparseArrays.SparseMatrixCSC{Tv, Ti}(m::AnyAbstractSparseMatrix) where {Tv, Ti}
    m′ = SparseMatrixCSC{Tv, Ti}(undef, size(m))
    for I in eachstoredindex(m)
        m′[I] = m[I]
    end
    return m′
end

function SparseArrayDOK(a::Base.ReshapedArray{<:Any, <:Any, <:AbstractSparseMatrixCSC})
    return SparseArrayDOK{eltype(a), ndims(a)}(a)
end
function SparseArrayDOK{T}(
        a::Base.ReshapedArray{<:Any, <:Any, <:AbstractSparseMatrixCSC}
    ) where {T}
    return SparseArrayDOK{T, ndims(a)}(a)
end
function SparseArrayDOK{T, N}(
        a::Base.ReshapedArray{<:Any, N, <:AbstractSparseMatrixCSC}
    ) where {T, N}
    a′ = SparseArrayDOK{T, N}(undef, size(a))
    for I in eachstoredindex(a)
        a′[I] = a[I]
    end
    return a′
end
