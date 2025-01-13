using SparseArraysBase: SparseArraysBase
using Test: @test, @testset
@testset "Test exports" begin
  exports = [
    :SparseArraysBase,
    :SparseArrayDOK,
    :SparseMatrixDOK,
    :SparseVectorDOK,
    :eachstoredindex,
    :isstored,
    :storedlength,
    :storedpairs,
    :storedvalues,
  ]
  @test issetequal(names(SparseArraysBase), exports)
end
