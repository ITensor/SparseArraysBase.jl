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
    :spzeros,
    :sprand,
    :sprand!,
  ]
  @test issetequal(names(SparseArraysBase), exports)
end
