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
using Test: @test, @test_throws, @testset

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
            if indexstyle == IndexCartesian()
                @test isstored(a, Tuple(I)..., 1)
            end
        end
    end
    @test eachstoredindex(a) == eachindex(a)
    # TODO: We should be specializing these for dense/strided arrays,
    # probably we can have a trait for that. It could be based
    # on the `ArrayLayouts.MemoryLayout`.
    @allowscalar @test storedvalues(a) == a
    @allowscalar @test storedpairs(a) == pairs(a)
    @allowscalar for I in eachindex(a)
        @test getstoredindex(a, I) == a[I]
        @test iszero(getunstoredindex(a, I))
    end
    @allowscalar for I in eachindex(IndexCartesian(), a)
        @test getstoredindex(a, I) == a[I]
        @test iszero(getunstoredindex(a, I))
    end

    n = 2
    a = @view dev(randn(elt, n, n))[1:2, 1]
    @test storedlength(a) == length(a)
    for indexstyle in (IndexLinear(), IndexCartesian())
        for I in eachindex(indexstyle, a)
            @test isstored(a, I)
        end
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
        @test_throws ErrorException setunstoredindex!(b, value, I...)
    end
end
