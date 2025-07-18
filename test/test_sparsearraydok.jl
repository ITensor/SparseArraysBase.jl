using Adapt: adapt
using ArrayLayouts: zero!
using Dictionaries: Dictionary
using FillArrays: Zeros
using JLArrays: JLArray, @allowscalar
using SparseArraysBase:
  SparseArraysBase,
  SparseArrayDOK,
  SparseMatrixDOK,
  eachstoredindex,
  getstoredindex,
  getunstoredindex,
  isstored,
  setstoredindex!,
  setunstoredindex!,
  sparse,
  sparserand,
  sparsezeros,
  storedlength,
  storedpairs,
  storedvalues
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset

elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
# arrayts = (Array, JLArray)
arrayts = (Array,)
@testset "SparseArrayDOK (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts,
  elt in elts

  dev(x) = adapt(arrayt, x)

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  @test a isa SparseArrayDOK{elt,2}
  @test size(a) == (2, 2)
  @test a[1] == 0
  @test a[1, 1] == 0
  @test a[1, 1, 1] == 0
  @test a[3] == 12
  @test a[1, 2] == 12
  @test a[1, 2, 1] == 12
  @test storedlength(a) == 1
  @test_throws BoundsError a[5]
  @test_throws BoundsError a[1, 3]

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[3] = 12
  for b in (similar(a, Float32, (3, 3)), similar(a, Float32, Base.OneTo.((3, 3))))
    @test b isa SparseArrayDOK{Float32,2}
    @test b == zeros(Float32, 3, 3)
    @test size(b) == (3, 3)
  end

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = similar(a)
  bc = Broadcast.Broadcasted(x -> 2x, (a,))
  copyto!(b, bc)
  @test b isa SparseArrayDOK{elt,2}
  @test b == [0 24; 0 0]
  @test storedlength(b) == 1

  # isstored
  a = SparseArrayDOK{elt}(undef, 4, 4)
  a[2, 3] = 23
  for (I, i) in zip(CartesianIndices(a), LinearIndices(a))
    if I == CartesianIndex(2, 3)
      @test isstored(a, I)
      @test isstored(a, Tuple(I)...)
      @test isstored(a, i)
    else
      @test !isstored(a, I)
      @test !isstored(a, Tuple(I)...)
      @test !isstored(a, i)
    end
  end

  # isstored SubArray
  a′ = SparseArrayDOK{elt}(undef, 4, 4)
  a′[2, 3] = 23
  a = @view a′[2:3, 2:3]
  for I in CartesianIndices(a)
    if I == CartesianIndex(1, 2)
      @test isstored(a, I)
      @test isstored(a, Tuple(I)...)
    else
      @test !isstored(a, I)
      @test !isstored(a, Tuple(I)...)
    end
  end

  # vector
  a = SparseArrayDOK{elt}(undef, 2)
  a[2] = 12
  @test b[1] == 0
  @test a[2] == 12
  @test storedlength(a) == 1

  a = SparseArrayDOK{elt}(undef, 3, 3, 3)
  a[1, 2, 3] = 123
  b = permutedims(a, (2, 3, 1))
  @test b isa SparseArrayDOK{elt,3}
  @test b[2, 3, 1] == 123
  @test storedlength(b) == 1
  @test b[1] == 0
  @test b[LinearIndices(b)[2, 3, 1]] == 123

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = copy(a')
  @test b isa SparseArrayDOK{elt,2}
  @test b == [0 0; 12 0]
  @test storedlength(b) == 1

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = map(x -> 2x, a)
  @test b isa SparseArrayDOK{elt,2}
  @test b == [0 24; 0 0]
  @test storedlength(b) == 1

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = a * a'
  @test b isa SparseArrayDOK{elt,2}
  @test b == [144 0; 0 0]
  @test storedlength(b) == 1

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = a .+ 2 .* a'
  @test b isa SparseArrayDOK{elt,2}
  @test b == [0 12; 24 0]
  @test storedlength(b) == 2

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = a[1:2, 2]
  @test b isa SparseArrayDOK{elt,1}
  @test b == [12, 0]
  @test storedlength(b) == 1

  a = SparseArrayDOK{elt}(undef, 2, 2)
  @test iszero(a)
  a[2, 1] = 21
  a[1, 2] = 12
  @test !iszero(a)
  @test isreal(a)
  @test sum(a) == 33
  @test mapreduce(x -> 2x, +, a) == 66

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = similar(a)
  copyto!(b, a)
  @test b isa SparseArrayDOK{elt,2}
  @test b == a
  @test b[1, 2] == 12
  @test storedlength(b) == 1

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a .= 2
  @test storedlength(a) == length(a)
  for I in eachindex(a)
    @test a[I] == 2
  end

  a = SparseArrayDOK{elt}(undef, 2, 2)
  fill!(a, 2)
  @test storedlength(a) == length(a)
  for I in eachindex(a)
    @test a[I] == 2
  end

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  zero!(a)
  @test iszero(a)
  @test iszero(storedlength(a))

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = zero(a)
  @test b isa SparseArrayDOK{elt,2}
  @test iszero(b)
  @test iszero(storedlength(b))

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = SparseArrayDOK{elt}(undef, 4, 4)
  b[2:3, 2:3] .= a
  @test isone(storedlength(b))
  @test b[2, 3] == 12

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = SparseArrayDOK{elt}(undef, 4, 4)
  b[2:3, 2:3] = a
  @test isone(storedlength(b))
  @test b[2, 3] == 12

  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 2] = 12
  b = SparseArrayDOK{elt}(undef, 4, 4)
  c = @view b[2:3, 2:3]
  c .= a
  @test isone(storedlength(b))
  @test b[2, 3] == 12

  a1 = SparseArrayDOK{elt}(undef, 2, 2)
  a1[1, 2] = 12
  a2 = SparseArrayDOK{elt}(undef, 2, 2)
  a2[2, 1] = 21
  b = cat(a1, a2; dims=(1, 2))
  @test b isa SparseArrayDOK{elt,2}
  @test storedlength(b) == 2
  @test b[1, 2] == 12
  @test b[4, 3] == 21

  if elt === Float64
    # Printing
    # Not testing other element types since they change the
    # spacing so it isn't easy to make the test general.
    a = SparseArrayDOK{elt}(undef, 2, 2)
    a[1, 2] = 12
    @test sprint(show, "text/plain", a) ==
      "$(summary(a)):\n  ⋅   $(eltype(a)(12))\n  ⋅     ⋅ "

    a = SparseArrayDOK{elt}(undef, 2)
    a[1] = 1
    @test sprint(show, "text/plain", a) == "$(summary(a)):\n $(eltype(a)(1))\n  ⋅ "
  end

  # Regression test for:
  # https://github.com/ITensor/SparseArraysBase.jl/issues/19
  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 1] = 1
  a .*= 2
  @test a == [2 0; 0 0]
  @test storedlength(a) == 1

  # Test aliasing behavior.
  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 1] = 11
  a[1, 2] = 12
  a[2, 2] = 22
  c1 = @view a[:, 1]
  r1 = @view a[1, :]
  r1 .= c1
  @test c1 == [11, 0]
  @test storedlength(c1) == 1
  @test r1 == [11, 0]
  @test storedlength(r1) == 2
  @test a == [11 0; 0 22]
  @test storedlength(a) == 3

  # Test aliasing behavior.
  a = SparseArrayDOK{elt}(undef, 2, 2)
  a[1, 1] = 11
  a[1, 2] = 12
  a[2, 2] = 22
  c1 = @view a[:, 1]
  r1 = @view a[1, :]
  c1 .= r1
  @test c1 == [11, 12]
  @test storedlength(c1) == 2
  @test r1 == [11, 12]
  @test storedlength(r1) == 2
  @test a == [11 12; 12 22]
  @test storedlength(a) == 4

  for d in (
    Dict([CartesianIndex(1, 2) => elt(12), CartesianIndex(2, 1) => elt(21)]),
    Dictionary([CartesianIndex(1, 2), CartesianIndex(2, 1)], [elt(12), elt(21)]),
  )
    for a in (
      sparse(d, 2, 2),
      sparse(d, Zeros{elt}(2, 2)),
      sparse(d, (2, 2)),
      # Determine the size automatically.
      sparse(d),
    )
      @test !iszero(a)
      @test iszero(a[1, 1])
      @test a[2, 1] == elt(21)
      @test a[1, 2] == elt(12)
      @test iszero(a[2, 2])
      @test size(a) == (2, 2)
      @test storedlength(a) == 2
      @test eltype(a) === elt
      @test a isa SparseMatrixDOK{elt}
    end
  end

  for (a, elt′) in (
    (sparsezeros(elt, 2, 2), elt),
    (sparsezeros(elt, Zeros{elt}(2, 2)), elt),
    (sparsezeros(elt, (2, 2)), elt),
    (sparsezeros(2, 2), Float64),
    (sparsezeros(Zeros{Float64}(2, 2)), Float64),
    (sparsezeros((2, 2)), Float64),
  )
    @test iszero(a)
    @test size(a) == (2, 2)
    @test storedlength(a) == 0
    @test eltype(a) === elt′
    @test a isa SparseMatrixDOK{elt′}
  end

  rng = StableRNG(123)
  for (a, elt′) in (
    (sparserand(rng, elt, 20, 20; density=0.5), elt),
    (sparserand(rng, elt, (20, 20); density=0.5), elt),
    (sparserand(rng, 20, 20; density=0.5), Float64),
    (sparserand(rng, (20, 20); density=0.5), Float64),
  )
    @test !iszero(a)
    @test size(a) == (20, 20)
    @test !iszero(storedlength(a))
    @test eltype(a) === elt′
    @test a isa SparseMatrixDOK{elt′}
  end
end
