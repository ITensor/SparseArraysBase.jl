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
    :storedlength,
    :storedpairs,
    :storedvalues,
    :sparsezeros,
    :sparserand,
    :sparserand!,
  ]
  @test issetequal(names(SparseArraysBase), exports)
end
