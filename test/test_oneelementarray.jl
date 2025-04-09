using SparseArraysBase:
  OneElementArray,
  OneElementMatrix,
  OneElementVector,
  eachstoredindex,
  isstored,
  oneelement,
  storedlength,
  storedpairs,
  storedvalues
using Test: @test, @test_broken, @testset

elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "OneElementArray (eltype=$elt)" for elt in elts
  for a in (
    OneElementArray((1, 2), (2, 2)),
    OneElementMatrix((1, 2), (2, 2)),
    OneElementArray((1, 2), Base.OneTo.((2, 2))),
    OneElementMatrix((1, 2), Base.OneTo.((2, 2))),
    OneElementArray(Base.OneTo(2) => 1, Base.OneTo(2) => 2),
    OneElementMatrix(Base.OneTo(2) => 1, Base.OneTo(2) => 2),
    oneelement((1, 2), (2, 2)),
    oneelement((1, 2), Base.OneTo.((2, 2))),
    oneelement(Base.OneTo(2) => 1, Base.OneTo(2) => 2),
  )
    @test a isa OneElementArray{Bool,2}
    @test a isa OneElementMatrix{Bool}
    @test eltype(a) === Bool
    @test size(a) == (2, 2)
    @test length(a) == 4
    @test axes(a) == Base.OneTo.((2, 2))
    @test a[1, 1] === zero(Bool)
    @test a[2, 1] === zero(Bool)
    @test a[1, 2] === one(Bool)
    @test a[2, 2] === zero(Bool)
    @test storedlength(a) == 1
    @test collect(eachstoredindex(a)) == [CartesianIndex(1, 2)]
    @test collect(storedpairs(a)) == [CartesianIndex(1, 2) => 1]
    @test collect(storedvalues(a)) == [1]
  end

  for a in (OneElementArray(1, 2), OneElementVector(1, 2))
    @test a isa OneElementArray{Bool,1}
    @test a isa OneElementVector{Bool}
    @test eltype(a) === Bool
    @test a[1] === one(Bool)
    @test a[2] === zero(Bool)
    @test storedlength(a) == 1
    @test collect(eachstoredindex(a)) == [CartesianIndex(1)]
    @test collect(storedpairs(a)) == [CartesianIndex(1) => 1]
    @test collect(storedvalues(a)) == [1]
  end

  a = OneElementArray()
  @test eltype(a) === Bool
  @test size(a) == ()
  @test a[] === one(Bool)

  a = OneElementArray{elt}()
  @test eltype(a) === elt
  @test size(a) == ()
  @test a[] === one(elt)

  for a in (
    OneElementArray{elt}((1, 2), (2, 2)),
    OneElementMatrix{elt}((1, 2), (2, 2)),
    OneElementArray(one(elt), (1, 2), (2, 2)),
    OneElementMatrix(one(elt), (1, 2), (2, 2)),
    OneElementArray{elt}((1, 2), Base.OneTo.((2, 2))),
    OneElementMatrix{elt}((1, 2), Base.OneTo.((2, 2))),
    OneElementArray(one(elt), (1, 2), Base.OneTo.((2, 2))),
    OneElementMatrix(one(elt), (1, 2), Base.OneTo.((2, 2))),
    OneElementArray{elt}(Base.OneTo(2) => 1, Base.OneTo(2) => 2),
    OneElementMatrix{elt}(Base.OneTo(2) => 1, Base.OneTo(2) => 2),
    OneElementArray(one(elt), Base.OneTo(2) => 1, Base.OneTo(2) => 2),
    OneElementMatrix(one(elt), Base.OneTo(2) => 1, Base.OneTo(2) => 2),
    oneelement(elt, (1, 2), (2, 2)),
    oneelement(one(elt), (1, 2), (2, 2)),
    oneelement(elt, (1, 2), Base.OneTo.((2, 2))),
    oneelement(one(elt), (1, 2), Base.OneTo.((2, 2))),
    oneelement(elt, Base.OneTo(2) => 1, Base.OneTo(2) => 2),
    oneelement(one(elt), Base.OneTo(2) => 1, Base.OneTo(2) => 2),
  )
    @test eltype(a) === elt
    @test a[1, 1] === zero(elt)
    @test a[2, 1] === zero(elt)
    @test a[1, 2] === one(elt)
    @test a[2, 2] === zero(elt)
  end

  a = OneElementArray{elt}((1, 2), (2, 2))
  b = 2a
  @test eltype(b) === elt
  @test storedlength(b) == 1
  # TODO: Need to preserve that it is a `OneElementArray`.
  # Currently falls back to constructing a `SparseArrayDOK`.
  @test_broken b isa OneElementMatrix{elt}
  @test b == 2 * Array(a)

  a1 = OneElementArray{elt}(2, (1, 2), (2, 2))
  a2 = OneElementArray{elt}(3, (2, 1), (2, 2))
  b = a1 * a2
  @test eltype(b) === elt
  @test b[1, 1] === elt(6)
  @test storedlength(b) == 1
  @test isstored(b, 1, 1)
  # TODO: Need to preserve that it is a `OneElementArray`.
  # Currently falls back to constructing a `SparseArrayDOK`.
  @test_broken b isa OneElementMatrix{elt}
  @test b == Array(a1) * Array(a2)
end
