using Adapt: adapt
using JLArrays: JLArray, @allowscalar
using SparseArraysBase:
  SparseArraysBase,
  eachstoredindex,
  getstoredindex,
  getunstoredindex,
  isstored,
  setstoredindex!,
  setunstoredindex!,
  storedlength,
  storedpairs,
  storedvalues
using Test: @test, @testset

elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
arrayts = (Array, JLArray)
@testset "SparseArraysBase (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts,
  elt in elts

  dev(x) = adapt(arrayt, x)

  n = 2
  a = dev(randn(elt, n, n))
  @test storedlength(a) == length(a)
  for indexstyle in (IndexLinear(), IndexCartesian())
    for I in eachindex(indexstyle, a)
      @test isstored(a, I)
    end
  end
  @test eachstoredindex(a) == eachindex(a)
  # TODO: We should be specializing these for dense/strided arrays,
  # probably we can have a trait for that. It could be based
  # on the `ArrayLayouts.MemoryLayout`.
  @allowscalar @test storedvalues(a) == vec(a)
  @allowscalar @test storedpairs(a) == collect(pairs(vec(a)))
  @allowscalar for I in eachindex(a)
    @test getstoredindex(a, I) == a[I]
    @test iszero(getunstoredindex(a, I))
  end
  @allowscalar for I in eachindex(IndexCartesian(), a)
    @test getstoredindex(a, I) == a[I]
    @test iszero(getunstoredindex(a, I))
  end

  a = dev(randn(elt, n, n))
  for I in ((1, 2), (CartesianIndex(1, 2),))
    b = copy(a)
    value = randn(elt)
    @allowscalar setstoredindex!(b, value, I...)
    @allowscalar b[I...] == value
  end

  # TODO: Should `setunstoredindex!` error by default
  # if the value at that index is already stored?
  a = dev(randn(elt, n, n))
  for I in ((1, 2), (CartesianIndex(1, 2),))
    b = copy(a)
    value = randn(elt)
    @allowscalar setunstoredindex!(b, value, I...)
    @allowscalar b[I...] == value
  end
end
