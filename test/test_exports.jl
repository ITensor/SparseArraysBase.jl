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
    :oneelement,
    :sparserand,
    :sparserand!,
    :sparsezeros,
    :storedlength,
    :storedpairs,
    :storedvalues,
  ]
  @test issetequal(names(SparseArraysBase), exports)
end
