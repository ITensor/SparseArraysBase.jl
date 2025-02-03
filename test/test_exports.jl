using SparseArraysBase: SparseArraysBase
using Test: @test, @testset
@testset "Test exports" begin
  exports = [
    :SparseArraysBase,
    :SparseArrayDOK,
    :SparseMatrixDOK,
    :SparseVectorDOK,
    :OneElementArray,
    :OneElementMatrix,
    :OneElementVector,
    :eachstoredindex,
    :isstored,
    :oneelementarray,
    :storedlength,
    :storedpairs,
    :storedvalues,
  ]
  @test issetequal(names(SparseArraysBase), exports)
end
