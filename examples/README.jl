# # SparseArraysBase.jl
# 
# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/SparseArraysBase.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/SparseArraysBase.jl/dev/)
# [![Build Status](https://github.com/ITensor/SparseArraysBase.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/SparseArraysBase.jl/actions/workflows/Tests.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/ITensor/SparseArraysBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/SparseArraysBase.jl)
# [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
# [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# ## Installation instructions

# This package resides in the `ITensor/ITensorRegistry` local registry.
# In order to install, simply add that registry through your package manager.
# This step is only required once.
#=
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
=#
# or:
#=
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
=#
# if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

# Then, the package can be added as usual through the package manager:

#=
```julia
julia> Pkg.add("SparseArraysBase")
```
=#

# ## Examples

using SparseArraysBase:
  SparseArrayDOK,
  eachstoredindex,
  getstoredindex,
  getunstoredindex,
  isstored,
  setstoredindex!,
  setunstoredindex!,
  storedlength,
  storedpairs,
  storedvalues
using Test: @test, @test_throws

a = SparseArrayDOK{Float64}(2, 2)

# AbstractArray interface:

a[1, 2] = 12
@test a == [0 12; 0 0]
@test a[1, 1] == 0
@test a[2, 1] == 0
@test a[1, 2] == 12
@test a[2, 2] == 0

# SparseArraysBase interface:

@test issetequal(eachstoredindex(a), [CartesianIndex(1, 2)])
@test getstoredindex(a, 1, 2) == 12
@test_throws KeyError getstoredindex(a, 1, 1)
@test getunstoredindex(a, 1, 1) == 0
@test getunstoredindex(a, 1, 2) == 0
@test !isstored(a, 1, 1)
@test isstored(a, 1, 2)
@test setstoredindex!(copy(a), 21, 1, 2) == [0 21; 0 0]
@test_throws KeyError setstoredindex!(copy(a), 21, 2, 1)
@test setunstoredindex!(copy(a), 21, 1, 2) == [0 21; 0 0]
@test storedlength(a) == 1
@test issetequal(storedpairs(a), [CartesianIndex(1, 2) => 12])
@test issetequal(storedvalues(a), [12])

# AbstractArray functionality:

b = a .+ 2 .* a'
@test b == [0 12; 24 0]
@test storedlength(b) == 2
@test b isa SparseArrayDOK{Float64}

a * a'
