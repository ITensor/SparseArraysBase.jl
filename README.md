# SparseArraysBase.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/SparseArraysBase.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/SparseArraysBase.jl/dev/)
[![Build Status](https://github.com/ITensor/SparseArraysBase.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/SparseArraysBase.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/SparseArraysBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/SparseArraysBase.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Installation instructions

This package resides in the `ITensor/ITensorRegistry` local registry.
In order to install, simply add that registry through your package manager.
This step is only required once.
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
or:
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

Then, the package can be added as usual through the package manager:

```julia
julia> Pkg.add("SparseArraysBase")
```

## Examples

````julia
using SparseArraysBase:
  SparseArrayDOK,
  SparseMatrixDOK,
  SparseVectorDOK,
  eachstoredindex,
  getstoredindex,
  getunstoredindex,
  isstored,
  setstoredindex!,
  setunstoredindex!,
  storedlength,
  storedpairs,
  storedvalues,
  zero!
using Test: @test, @test_throws

a = SparseArrayDOK{Float64}(2, 2)
````

AbstractArray interface:

````julia
@test iszero(a)
@test iszero(sum(a))
@test iszero(storedlength(a))

a[1, 2] = 12
@test a == [0 12; 0 0]
@test a[1, 1] == 0
@test a[2, 1] == 0
@test a[1, 2] == 12
@test a[2, 2] == 0
````

SparseArraysBase interface:

````julia
using Dictionaries: IndexError
@test issetequal(eachstoredindex(a), [CartesianIndex(1, 2)])
@test getstoredindex(a, 1, 2) == 12
@test_throws IndexError getstoredindex(a, 1, 1)
@test getunstoredindex(a, 1, 1) == 0
@test getunstoredindex(a, 1, 2) == 0
@test !isstored(a, 1, 1)
@test isstored(a, 1, 2)
@test setstoredindex!(copy(a), 21, 1, 2) == [0 21; 0 0]
@test_throws IndexError setstoredindex!(copy(a), 21, 2, 1)
@test setunstoredindex!(copy(a), 21, 1, 2) == [0 21; 0 0]
@test storedlength(a) == 1
@test issetequal(storedpairs(a), [CartesianIndex(1, 2) => 12])
@test issetequal(storedvalues(a), [12])
@test sum(a) == 12
@test isreal(a)
@test !iszero(a)
@test mapreduce(x -> 2x, +, a) == 24
````

AbstractArray functionality:

````julia
b = a .+ 2 .* a'
@test b isa SparseMatrixDOK{Float64}
@test b == [0 12; 24 0]
@test storedlength(b) == 2
@test sum(b) == 36
@test isreal(b)
@test !iszero(b)
@test mapreduce(x -> 2x, +, b) == 72

b = permutedims(a, (2, 1))
@test b isa SparseMatrixDOK{Float64}
@test b[1, 1] == a[1, 1]
@test b[2, 1] == a[1, 2]
@test b[1, 2] == a[2, 1]
@test b[2, 2] == a[2, 2]

b = a * a'
@test b isa SparseMatrixDOK{Float64}
@test b == [144 0; 0 0]
@test storedlength(b) == 1
````

Second column.

````julia
b = a[1:2, 2]
@test b isa SparseVectorDOK{Float64}
@test b == [12, 0]
@test storedlength(b) == 1

a = SparseArrayDOK{Float64}(2, 2)
a .= 2
for I in eachindex(a)
  @test a[I] == 2
end
@test storedlength(a) == length(a)

a = SparseArrayDOK{Float64}(2, 2)
fill!(a, 2)
for I in eachindex(a)
  @test a[I] == 2
end
@test storedlength(a) == length(a)

a = SparseArrayDOK{Float64}(2, 2)
fill!(a, 0)
@test iszero(a)
@test iszero(storedlength(a))

a = SparseArrayDOK{Float64}(2, 2)
a[1, 2] = 12
zero!(a)
@test iszero(a)
@test iszero(storedlength(a))

a = SparseArrayDOK{Float64}(2, 2)
a[1, 2] = 12
b = zero(a)
@test iszero(b)
@test iszero(storedlength(b))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

