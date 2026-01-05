using Base: @_propagate_inbounds_meta
using FunctionImplementations: Implementation, style

# Indexing interface
# ------------------
# these definitions are not using @derive since we need the @inline annotation
# to correctly deal with boundschecks and @inbounds

"""
    getstoredindex(A::AbstractArray, I...) -> eltype(A)

Obtain `getindex(A, I...)` with the guarantee that there is a stored entry at that location.

Similar to `Base.getindex`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline function getstoredindex(A::AbstractArray, I...)
    return style(A)(getstoredindex)(A, I...)
end

"""
    getunstoredindex(A::AbstractArray, I...) -> eltype(A)

Obtain the value that would be returned by `getindex(A, I...)` when there is no stored entry
at that location.
By default, this takes an explicit copy of the `getindex` implementation to mimick a newly
instantiated object.

Similar to `Base.getindex`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline function getunstoredindex(A::AbstractArray, I...)
    return style(A)(getunstoredindex)(A, I...)
end

"""
    isstored(A::AbstractArray, I...) -> Bool

Check if the array `A` has a stored entry at the location specified by indices `I...`.
For generic array types this defaults to `true` whenever the indices are inbounds, but
sparse array types might overload this function when appropriate.

Similar to `Base.getindex`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline function isstored(A::AbstractArray, I...)
    return style(A)(isstored)(A, I...)
end

"""
    setstoredindex!(A::AbstractArray, v, I...) -> A

`setindex!(A, v, I...)` with the guarantee that there is a stored entry at the given location.

Similar to `Base.setindex!`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline function setstoredindex!(A::AbstractArray, v, I...)
    return style(A)(setstoredindex!)(A, v, I...)
end

"""
    setunstoredindex!(A::AbstractArray, v, I...) -> A

`setindex!(A, v, I...)` with the guarantee that there is no stored entry at the given location.

Similar to `Base.setindex!`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline function setunstoredindex!(A::AbstractArray, v, I...)
    return style(A)(setunstoredindex!)(A, v, I...)
end

# Indices interface
# -----------------
"""
    eachstoredindex(A::AbstractArray...)
    eachstoredindex(style::IndexStyle, A::AbstractArray...)

An iterable over all indices of the stored values.
For multiple arrays, the iterable contains all indices where at least one input has a stored value.
The type of indices can be controlled through `style`, which will default to a compatible style for all
inputs.

The order of the iterable is not guaranteed to be fixed or sorted, and should not be assumed
to be the same as [`storedvalues`](@ref).

See also [`storedvalues`](@ref), [`storedpairs`](@ref) and [`storedlength`](@ref).
"""
function eachstoredindex end

"""
    storedlength(A::AbstractArray) -> Int

The number of values that are currently being stored.
"""
function storedlength end

"""
    storedpairs(A::AbstractArray) -> (k, v)...

An iterable over all stored indices and their corresponding stored values.
The indices are compatible with `IndexStyle(A)`.

The order of the iterable is not guaranteed to be fixed or sorted.
See also [`eachstoredindex`](@ref) and [`storedvalues`](@ref).
"""
function storedpairs end

"""
    storedvalues(A::AbstractArray) -> v...

An iterable over all stored values.

The order of the iterable is not guaranteed to be fixed or sorted, and should not be assumed
to be the same as [`eachstoredindex`](@ref).
"""
function storedvalues end

## @derive (T = AbstractArray,) begin
##     SparseArraysBase.eachstoredindex(::T...)
##     SparseArraysBase.eachstoredindex(::IndexStyle, ::T...)
##     SparseArraysBase.storedlength(::T)
##     SparseArraysBase.storedpairs(::T)
##     SparseArraysBase.storedvalues(::T)
## end

# canonical indexing
# ------------------
# ensure functions only have to be defined in terms of a single canonical f:
#   f(::AbstractArray, I::Int) if IndexLinear
#   f(::AbstractArray{<:Any,N}, I::Vararg{Int,N}) if IndexCartesian

for f in (:isstored, :getunstoredindex, :getstoredindex)
    _f = Symbol(:_, f)
    error_if_canonical = Symbol(:error_if_canonical_, f)
    @eval begin
        function (::Implementation{typeof($f)})(A::AbstractArray, I...)
            @_propagate_inbounds_meta
            style = IndexStyle(A)
            $error_if_canonical(style, A, I...)
            return $_f(style, A, Base.to_indices(A, I)...)
        end

        # linear indexing
        @inline $_f(::IndexLinear, A::AbstractVector, i::Int) = $f(A, i)
        @inline $_f(::IndexLinear, A::AbstractArray, i::Int) = $f(A, i)
        @inline function $_f(::IndexLinear, A::AbstractArray, I::Vararg{Int, M}) where {M}
            @boundscheck checkbounds(A, I...)
            return @inbounds $f(A, Base._to_linear_index(A, I...))
        end

        # cartesian indexing
        @inline function $_f(::IndexCartesian, A::AbstractArray, I::Vararg{Int, M}) where {M}
            @boundscheck checkbounds(A, I...)
            return @inbounds $f(A, Base._to_subscript_indices(A, I...)...)
        end
        @inline function $_f(
                ::IndexCartesian, A::AbstractArray{<:Any, N}, I::Vararg{Int, N}
            ) where {N}
            return $f(A, I...)
        end

        # errors
        $_f(::IndexStyle, A::AbstractArray, I...) = error(
            "`$($f)` for $("$(typeof(A))") with types $("$(typeof(I))") is not supported"
        )

        $error_if_canonical(::IndexLinear, A::AbstractArray, ::Int) = throw(
            Base.CanonicalIndexError("$($f)", typeof(A))
        )
        $error_if_canonical(::IndexCartesian, A::AbstractArray{<:Any, N}, ::Vararg{Int, N}) where {N} = throw(
            Base.CanonicalIndexError("$($f)", typeof(A))
        )
        $error_if_canonical(::IndexStyle, A::AbstractArray, ::Any...) = nothing
    end
end

for f! in (:setstoredindex!, :setunstoredindex!)
    _f! = Symbol(:_, f!)
    error_if_canonical = Symbol(:error_if_canonical_, f!)
    @eval begin
        function (::Implementation{typeof($f!)})(A::AbstractArray, v, I...)
            @_propagate_inbounds_meta
            style = IndexStyle(A)
            $error_if_canonical(style, A, I...)
            return $_f!(style, A, v, Base.to_indices(A, I)...)
        end

        # linear indexing
        @inline $_f!(::IndexLinear, A::AbstractVector, v, i::Int) = $f!(A, v, i)
        @inline $_f!(::IndexLinear, A::AbstractArray, v, i::Int) = $f!(A, v, i)
        @inline function $_f!(::IndexLinear, A::AbstractArray, v, I::Vararg{Int, M}) where {M}
            @boundscheck checkbounds(A, I...)
            return @inbounds $f!(A, v, Base._to_linear_index(A, I...))
        end

        # cartesian indexing
        @inline function $_f!(::IndexCartesian, A::AbstractArray, v, I::Vararg{Int, M}) where {M}
            @boundscheck checkbounds(A, I...)
            return @inbounds $f!(A, v, Base._to_subscript_indices(A, I...)...)
        end
        @inline function $_f!(
                ::IndexCartesian, A::AbstractArray{<:Any, N}, v, I::Vararg{Int, N}
            ) where {N}
            return $f!(A, v, I...)
        end

        # errors
        $_f!(::IndexStyle, A::AbstractArray, I...) = error(
            "`$f!` for $("$(typeof(A))") with types $("$(typeof(I))") is not supported"
        )

        $error_if_canonical(::IndexLinear, A::AbstractArray, ::Int) = throw(
            Base.CanonicalIndexError("$($(string(f!)))", typeof(A))
        )
        $error_if_canonical(::IndexCartesian, A::AbstractArray{<:Any, N}, ::Vararg{Int, N}) where {N} = throw(
            Base.CanonicalIndexError("$($f!)", typeof(A))
        )
        $error_if_canonical(::IndexStyle, A::AbstractArray, ::Any...) = nothing
    end
end

# AbstractArrayInterface fallback definitions
# -------------------------------------------
function isstored(A::AbstractArray, i::Int, I::Int...)
    @inline
    @boundscheck checkbounds(A, i, I...)
    return true
end

function getunstoredindex(A::AbstractArray, I::Int...)
    @inline
    @boundscheck checkbounds(A, I...)
    return zero(eltype(A))
end
function getstoredindex(A::AbstractArray, I::Int...)
    @inline
    return getindex(A, I...)
end

function setstoredindex!(A::AbstractArray, v, I::Int...)
    @inline
    return setindex!(A, v, I...)
end
setunstoredindex!(A::AbstractArray, v, I::Int...) = error(
    "setunstoredindex! for $(typeof(A)) is not supported"
)

eachstoredindex(A::AbstractArray, B::AbstractArray...) = eachstoredindex(
    IndexStyle(A, B...), A, B...
)
eachstoredindex(style::IndexStyle, A::AbstractArray, B::AbstractArray...) = eachindex(
    style, A, B...
)

storedvalues(A::AbstractArray) = values(A)
storedpairs(A::AbstractArray) = pairs(A)
storedlength(A::AbstractArray) = length(storedvalues(A))

# SparseArrayInterface implementations
# ------------------------------------
# canonical errors are moved to `isstored`, `getstoredindex` and `getunstoredindex`
# so no errors at this level by defining both IndexLinear and IndexCartesian
const getindex_sparse = sparse_style(getindex)
function getindex_sparse(
        A::AbstractArray{<:Any, N}, I::Vararg{Int, N}
    ) where {N}
    @_propagate_inbounds_meta
    @boundscheck checkbounds(A, I...) # generally isstored requires bounds checking
    return @inbounds isstored(A, I...) ? getstoredindex(A, I...) : getunstoredindex(A, I...)
end
function getindex_sparse(A::AbstractArray, I::Int)
    @_propagate_inbounds_meta
    @boundscheck checkbounds(A, I)
    return @inbounds isstored(A, I) ? getstoredindex(A, I) : getunstoredindex(A, I)
end
# disambiguate vectors
function getindex_sparse(A::AbstractVector, I::Int)
    @_propagate_inbounds_meta
    @boundscheck checkbounds(A, I)
    return @inbounds isstored(A, I) ? getstoredindex(A, I) : getunstoredindex(A, I)
end

const setindex!_sparse = sparse_style(setindex!)
function setindex!_sparse(
        A::AbstractArray{<:Any, N}, v, I::Vararg{Int, N}
    ) where {N}
    @_propagate_inbounds_meta
    @boundscheck checkbounds(A, I...)
    return @inbounds if isstored(A, I...)
        setstoredindex!(A, v, I...)
    else
        setunstoredindex!(A, v, I...)
    end
end
function setindex!_sparse(
        A::AbstractArray, v, I::Int
    )
    @_propagate_inbounds_meta
    @boundscheck checkbounds(A, I)
    return @inbounds if isstored(A, I)
        setstoredindex!(A, v, I)
    else
        setunstoredindex!(A, v, I)
    end
end
# disambiguate vectors
function setindex!_sparse(
        A::AbstractVector, v, I::Int
    )
    @_propagate_inbounds_meta
    @boundscheck checkbounds(A, I)
    return @inbounds if isstored(A, I)
        setstoredindex!(A, v, I)
    else
        setunstoredindex!(A, v, I)
    end
end

@noinline function error_if_canonical_eachstoredindex(style::IndexStyle, A::AbstractArray)
    style === IndexStyle(A) && throw(Base.CanonicalIndexError("eachstoredindex", typeof(A)))
    return nothing
end

# required: one implementation for canonical index style
const eachstoredindex_sparse = sparse_style(eachstoredindex)
function eachstoredindex_sparse(
        style::IndexStyle, A::AbstractArray
    )
    error_if_canonical_eachstoredindex(style, A)
    inds = eachstoredindex(A)
    if style === IndexCartesian()
        eltype(inds) === CartesianIndex{ndims(A)} && return inds
        return map(Base.Fix1(Base.getindex, CartesianIndices(A)), inds)
    elseif style === IndexLinear()
        eltype(inds) === Int && return inds
        return map(Base.Fix1(Base.getindex, LinearIndices(A)), inds)
    else
        error(lazy"unkown index style $style")
    end
end

# derived but may be specialized:
function eachstoredindex_sparse(
        style::IndexStyle, A::AbstractArray, B::AbstractArray...
    )
    return union(map(Base.Fix1(eachstoredindex, style), (A, B...))...)
end

const storedvalues_sparse = sparse_style(storedvalues)
storedvalues_sparse(A::AbstractArray) = StoredValues(A)

# default implementation is a bit tricky here: we don't know if this is the "canonical"
# implementation, so we check this and otherwise map back to `_isstored` to canonicalize the
# indices
const isstored_sparse = sparse_style(isstored)
function isstored_sparse(A::AbstractArray, I::Int...)
    @_propagate_inbounds_meta
    style = IndexStyle(A)
    # canonical linear indexing
    if style == IndexLinear() && length(I) == 1
        @boundscheck checkbounds(A, I...)
        return only(I) in eachstoredindex(style, A)
    end

    # canonical cartesian indexing
    if style == IndexCartesian() && length(I) == ndims(A)
        @boundscheck checkbounds(A, I...)
        return CartesianIndex(I...) in eachstoredindex(style, A)
    end

    # non-canonical indexing
    return _isstored(style, A, Base.to_indices(A, I)...)
end

const getunstoredindex_sparse = sparse_style(getunstoredindex)
function getunstoredindex_sparse(
        A::AbstractArray, I::Int...
    )
    @_propagate_inbounds_meta
    style = IndexStyle(A)

    # canonical linear indexing
    if style == IndexLinear() && length(I) == 1
        @boundscheck checkbounds(A, I...)
        return zero(eltype(A))
    end

    # canonical cartesian indexing
    if style == IndexCartesian() && length(I) == ndims(A)
        @boundscheck checkbounds(A, I...)
        return zero(eltype(A))
    end

    # non-canonical indexing
    return _getunstoredindex(style, A, Base.to_indices(A, I)...)
end

const getstoredindex_sparse = sparse_style(getstoredindex)
function getstoredindex_sparse(
        A::AbstractArray, I::Int...
    )
    @_propagate_inbounds_meta
    style = IndexStyle(A)
    error_if_canonical_getstoredindex(style, A, I...)
    return _getstoredindex(style, A, Base.to_indices(A, I)...)
end

for f! in (:setstoredindex!, :setunstoredindex!)
    _f! = Symbol(:_, f!)
    error_if_canonical_setstoredindex = Symbol(:error_if_canonical_, f!)
    @eval begin
        function (::Implementation{typeof($f!)})(A::AbstractArray, v, I::Int...)
            @_propagate_inbounds_meta
            style = IndexStyle(A)
            $error_if_canonical_setstoredindex(style, A, I...)
            return $_f!(style, A, v, Base.to_indices(A, I)...)
        end
    end
end

const storedlength_sparse = sparse_style(storedlength)
storedlength_sparse(A::AbstractArray) = length(
    storedvalues(A)
)
const storedpairs_sparse = sparse_style(storedpairs)
function storedpairs_sparse(A::AbstractArray)
    return Iterators.map(I -> (I => A[I]), eachstoredindex(A))
end

#=
All sparse array interfaces are mapped through layout_getindex. (is this too opinionated?)

using ArrayLayouts getindex: this is a bit cumbersome because there already is a way to make
that work focused on types but here we want to focus on interfaces.
eg: ArrayLayouts.@layoutgetindex ArrayType
TODO: decide if we need the interface approach at all here
=#
for (Tr, Tc) in Iterators.product(
        Iterators.repeated((:Colon, :AbstractUnitRange, :AbstractVector, :Integer), 2)...
    )
    Tr === Tc === :Integer && continue
    @eval begin
        function getindex_sparse(
                A::AbstractMatrix, kr::$Tr, jr::$Tc
            )
            Base.@inline # needed to make boundschecks work
            return ArrayLayouts.layout_getindex(A, kr, jr)
        end
    end
end
