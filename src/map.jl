# zero-preserving Traits
# ----------------------
"""
    abstract type ZeroPreserving <: Function end

Holy Trait to indicate how a function interacts with abstract zero values:

- `StrongPreserving` : output is guaranteed to be zero if **any** input is.
- `WeakPreserving` : output is guaranteed to be zero if **all** inputs are.
- `NonPreserving` : no guarantees on output.

To attempt to automatically determine this, either `ZeroPreserving(f, A::AbstractArray...)` or
`ZeroPreserving(f, T::Type...)` can be used/overloaded.

!!! warning
    incorrectly registering a function to be zero-preserving will lead to silently wrong results.
"""
abstract type ZeroPreserving <: Function end

struct StrongPreserving{F} <: ZeroPreserving
  f::F
end
struct WeakPreserving{F} <: ZeroPreserving
  f::F
end
struct NonPreserving{F} <: ZeroPreserving
  f::F
end

# Backport: remove in 1.12
@static if !isdefined(Base, :haszero)
  _haszero(T::Type) = false
  _haszero(::Type{<:Number}) = true
else
  _haszero = Base.haszero
end

# warning: cannot automatically detect WeakPreserving since this would mean checking all values
function ZeroPreserving(f, A::AbstractArray, Bs::AbstractArray...)
  return ZeroPreserving(f, eltype(A), eltype.(Bs)...)
end
# TODO: the following might not properly specialize on the types
# TODO: non-concrete element types
function ZeroPreserving(f, T::Type, Ts::Type...)
  if all(_haszero, (T, Ts...))
    return iszero(f(zero(T), zero.(Ts)...)) ? WeakPreserving(f) : NonPreserving(f)
  else
    return NonPreserving(f)
  end
end

const _WEAK_FUNCTIONS = (:+, :-)
for f in _WEAK_FUNCTIONS
  @eval begin
    ZeroPreserving(::typeof($f), ::Type{<:Number}, ::Type{<:Number}...) = WeakPreserving($f)
  end
end

const _STRONG_FUNCTIONS = (:*,)
for f in _STRONG_FUNCTIONS
  @eval begin
    ZeroPreserving(::typeof($f), ::Type{<:Number}, ::Type{<:Number}...) = StrongPreserving(
      $f
    )
  end
end

# map(!)
# ------
@interface I::AbstractSparseArrayInterface function Base.map(
  f, A::AbstractArray, Bs::AbstractArray...
)
  f_pres = ZeroPreserving(f, A, Bs...)
  return @interface I map(f_pres, A, Bs...)
end
@interface I::AbstractSparseArrayInterface function Base.map(
  f::ZeroPreserving, A::AbstractArray, Bs::AbstractArray...
)
  T = Base.Broadcast.combine_eltypes(f.f, (A, Bs...))
  C = similar(I, T, size(A))
  return @interface I map!(f, C, A, Bs...)
end

@interface I::AbstractSparseArrayInterface function Base.map!(
  f, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...
)
  f_pres = ZeroPreserving(f, A, Bs...)
  return @interface I map!(f_pres, C, A, Bs...)
end

@interface ::AbstractSparseArrayInterface function Base.map!(
  f::ZeroPreserving, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...
)
  checkshape(C, A, Bs...)
  unaliased = map(Base.Fix1(Base.unalias, C), (A, Bs...))

  if f isa StrongPreserving
    style = IndexStyle(C, unaliased...)
    inds = intersect(eachstoredindex.(Ref(style), unaliased)...)
    zero!(C)
  elseif f isa WeakPreserving
    style = IndexStyle(C, unaliased...)
    inds = union(eachstoredindex.(Ref(style), unaliased)...)
    zero!(C)
  elseif f isa NonPreserving
    inds = eachindex(C, unaliased...)
  else
    error(lazy"unknown zero-preserving type $(typeof(f))")
  end

  @inbounds for I in inds
    C[I] = f.f(ith_all(I, unaliased)...)
  end

  return C
end

# Derived functions
# -----------------
@interface I::AbstractSparseArrayInterface Base.copyto!(C::AbstractArray, A::AbstractArray) = @interface I map!(
  identity, C, A
)

# Utility functions
# -----------------
# shape check similar to checkbounds
checkshape(::Type{Bool}, A::AbstractArray) = true
checkshape(::Type{Bool}, A::AbstractArray, B::AbstractArray) = size(A) == size(B)
function checkshape(::Type{Bool}, A::AbstractArray, Bs::AbstractArray...)
  return allequal(size, (A, Bs...))
end

function checkshape(A::AbstractArray, Bs::AbstractArray...)
  return checkshape(Bool, A, Bs...) ||
         throw(DimensionMismatch("argument shapes must match"))
end

@inline ith_all(i, ::Tuple{}) = ()
function ith_all(i, as)
  @_propagate_inbounds_meta
  return (as[1][i], ith_all(i, Base.tail(as))...)
end
