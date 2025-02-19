using Base: @_propagate_inbounds_meta

# Indexing interface
# ------------------
# these definitions are not using @derive since we need the @inline annotation
# to correctly deal with boundschecks and @inbounds

"""
    getstoredindex(A::AbstractArray, I...) -> eltype(A)

Obtain `getindex(A, I...)` with the guarantee that there is a stored entry at that location.

Similar to `Base.getindex`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline getstoredindex(A::AbstractArray, I...) =
  @interface interface(A) getstoredindex(A, I...)

"""
    getunstoredindex(A::AbstractArray, I...) -> eltype(A)

Obtain `getindex(A, I...)` with the guarantee that there is no stored entry at that location.
By default, this takes an explicit copy of the `getindex` implementation to mimick a newly
instantiated object.

Similar to `Base.getindex`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline getunstoredindex(A::AbstractArray, I...) =
  @interface interface(A) getunstoredindex(A, I...)

"""
    isstored(A::AbstractArray, I...) -> Bool

Check if the array `A` has a stored entry at the location specified by indices `I...`.
For generic array types this defaults to `true` whenever the indices are inbounds, but
sparse array types might overload this function when appropriate.

Similar to `Base.getindex`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline isstored(A::AbstractArray, I...) = @interface interface(A) isstored(A, I...)

"""
    setstoredindex!(A::AbstractArray, v, I...) -> A

`setindex!(A, v, I...)` with the guarantee that there is a stored entry at the given location.

Similar to `Base.setindex!`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline setstoredindex!(A::AbstractArray, v, I...) =
  @interface interface(A) setstoredindex!(A, v, I...)

"""
    setunstoredindex!(A::AbstractArray, v, I...) -> A

`setindex!(A, v, I...)` with the guarantee that there is no stored entry at the given location.

Similar to `Base.setindex!`, new definitions should be in line with `IndexStyle(A)`.
"""
@inline setunstoredindex!(A::AbstractArray, v, I...) =
  @interface interface(A) setunstoredindex!(A, v, I...)

# Indices interface
# -----------------
"""
    eachstoredindex(A::AbstractArray...)
    eachstoredindex(style::IndexStyle, A::AbstractArray...)

An iterable over all indices of the stored values.
For multiple arrays, the iterable contains all indices where at least one input has a stored value.
The type of indices can be controlled through `style`, which will default to a compatible style for all
inputs.

The order of the iterable is not fixed, but for a single input it may be assumed to be compatible
with [`storedvalues`](@ref).

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

The order of the iterable is not fixed, but is compatible with [`eachstoredindex`](@ref).
"""
function storedpairs end

"""
    storedvalues(A::AbstractArray) -> v...

An iterable over all stored values.

The order of the iterable is not fixed, but is compatible with [`eachstoredindex`](@ref).
"""
function storedvalues end

@derive (T=AbstractArray,) begin
  eachstoredindex(::T...)
  eachstoredindex(::IndexStyle, ::T...)
  storedlength(::T)
  storedpairs(::T)
  storedvales(::T)
end

# canonical indexing
# ------------------
# ensure functions only have to be defined in terms of a single canonical f:
#   f(::AbstractArray, I::Int) if IndexLinear
#   f(::AbstractArray{<:Any,N}, I::Vararg{Int,N}) if IndexCartesian

for f in (:isstored, :getunstoredindex, :getstoredindex)
  _f = Symbol(:_, f)
  error_if_canonical = Symbol(:error_if_canonical_, f)
  @eval begin
    @interface ::AbstractArrayInterface function $f(A::AbstractArray, I...)
      @_propagate_inbounds_meta
      style = IndexStyle(A)
      $error_if_canonical(style, A, I...)
      return $_f(style, A, Base.to_indices(A, I...))
    end

    # linear indexing
    @inline $_f(::IndexLinear, A::AbstractVector, i::Int) = $f(A, i)
    @inline $_f(::IndexLinear, A::AbstractArray, i::Int) = $f(A, i)
    @inline function $_f(::IndexLinear, A::AbstractArray, I::Vararg{Int,M}) where {M}
      @boundscheck checkbounds(A, I...)
      return @inbounds $f(A, Base._to_linear_index(A, I...))
    end

    # cartesian indexing
    @inline function $_f(::IndexCartesian, A::AbstractArray, I::Vararg{Int,M}) where {M}
      @boundscheck checkbounds(A, I...)
      return @inbounds $f(A, Base._to_subscript_indices(A, I...)...)
    end
    @inline function $_f(
      ::IndexCartesian, A::AbstractArray{<:Any,N}, I::Vararg{Int,N}
    ) where {N}
      return $f(A, I...)
    end

    # errors
    $_f(::IndexStyle, A::AbstractArray, I...) =
      error("`$f` for $("$(typeof(A))") with types $("$(typeof(I))") is not supported")

    $error_if_canonical(::IndexLinear, A::AbstractArray, ::Int) =
      throw(Base.CanonicalIndexError("$f", typeof(A)))
    $error_if_canonical(
      ::IndexCartesian, A::AbstractArray{<:Any,N}, ::Vararg{Int,N}
    ) where {N} = throw(Base.CanonicalIndexError("$f", typeof(A)))
    $error_if_canonical(::IndexStyle, A::AbstractArray, ::Any...) = nothing
  end
end

for f! in (:setunstoredindex!, :setstoredindex!)
  _f! = Symbol(:_, f!)
  error_if_canonical = Symbol(:error_if_canonical_, f!)
  @eval begin
    @interface ::AbstractArrayInterface function $f!(A::AbstractArray, v, I...)
      @_propagate_inbounds_meta
      style = IndexStyle(A)
      $error_if_canonical(style, A, I...)
      return $_f!(A, v, I...)
    end

    # linear indexing
    @inline $_f!(::IndexLinear, A::AbstractVector, v, i::Int) = $f!(A, v, i)
    @inline $_f!(::IndexLinear, A::AbstractArray, v, i::Int) = $f!(A, v, i)
    @inline function $_f!(::IndexLinear, A::AbstractArray, v, I::Vararg{Int,M}) where {M}
      @boundscheck checkbounds(A, I...)
      return @inbounds $f!(A, v, Base._to_linear_index(A, I...))
    end

    # cartesian indexing
    @inline function $_f!(::IndexCartesian, A::AbstractArray, v, I::Vararg{Int,M}) where {M}
      @boundscheck checkbounds(A, I...)
      return @inbounds $f!(A, v, Base._to_subscript_indices(A, I...)...)
    end
    @inline function $_f!(A::AbstractArray{<:Any,N}, v, I::Vararg{Int,N}) where {N}
      return $f!(A, v, I...)
    end

    # errors
    $_f!(::IndexStyle, A::AbstractArray, v, I...) =
      error("`$f!` for $("$(typeof(A))") with types $("$(typeof(I))") is not supported")

    $error_if_canonical(::IndexLinear, A::AbstractArray, ::Int) =
      throw(Base.CanonicalIndexError("$f!", typeof(A)))
    $error_if_canonical(
      ::IndexCartesian, A::AbstractArray{<:Any,N}, ::Vararg{Int,N}
    ) where {N} = throw(Base.CanonicalIndexError("$f!", typeof(A)))
    $error_if_canonical(::IndexStyle, A::AbstractArray, ::Any...) = nothing
  end
end

# AbstractArrayInterface fallback definitions
# -------------------------------------------
@interface ::AbstractArrayInterface isstored(A::AbstractArray, I::Int...) = (
  @inline; @boundscheck checkbounds(A, I...); true
)
@interface ::AbstractArrayInterface function getunstoredindex(A::AbstractArray, I::Int...)
  @inline
  @boundscheck checkbounds(A, I...)
  return zero(eltype(A))
end
@interface ::AbstractArrayInterface getstoredindex(A::AbstractArray, I::Int...) = (
  @inline; getindex(A, I...)
)
@interface ::AbstractArrayInterface setstoredindex!(A::AbstractArray, I::Int...) = (
  @inline; setindex!(A, I...)
)
@interface ::AbstractArrayInterface setunstoredindex!(A::AbstractArray, v, I::Int...) =
  error("setunstoredindex! for $(typeof(A)) is not supported")

@interface ::AbstractArrayInterface eachstoredindex(A::AbstractArray, B::AbstractArray...) =
  eachstoredindex(IndexStyle(A, B...), A, B...)
@interface ::AbstractArrayInterface eachstoredindex(
  style::IndexStyle, A::AbstractArray, B::AbstractArray...
) = eachindex(style, A, B...)

@interface ::AbstractArrayInterface storedvalues(A::AbstractArray) = values(A)
@interface ::AbstractArrayInterface storedpairs(A::AbstractArray) = pairs(A)
@interface ::AbstractArrayInterface storedlength(A::AbstractArray) = length(A)
