# zero-preserving Traits
# ----------------------
"""
    abstract type ZeroPreserving end

Holy Trait to indicate how a function interacts with abstract zero values:

- `StrongPreserving` : output is guaranteed to be zero if **any** input is.
- `WeakPreserving` : output is guaranteed to be zero if **all** inputs are.
- `NonPreserving` : no guarantees on output.

To attempt to automatically determine this, either `ZeroPreserving(f, A::AbstractArray...)` or
`ZeroPreserving(f, T::Type...)` can be used/overloaded.

!!! warning
    incorrectly registering a function to be zero-preserving will lead to silently wrong results.
"""
abstract type ZeroPreserving end
struct StrongPreserving <: ZeroPreserving end
struct WeakPreserving <: ZeroPreserving end
struct NonPreserving <: ZeroPreserving end

# warning: cannot automatically detect WeakPreserving since this would mean checking all values
function ZeroPreserving(f, A::AbstractArray, Bs::AbstractArray...)
  return ZeroPreserving(f, eltype(A), eltype.(Bs)...)
end
# TODO: the following might not properly specialize on the types
# TODO: non-concrete element types
function ZeroPreserving(f, T::Type, Ts::Type...)
  return iszero(f(zero(T), zero.(Ts)...)) ? WeakPreserving() : NonPreserving()
end

const _WEAK_FUNCTIONS = (:+, :-)
for f in _WEAK_FUNCTIONS
  @eval begin
    ZeroPreserving(::typeof($f), ::AbstractArray, ::AbstractArray...) = WeakPreserving()
    ZeroPreserving(::typeof($f), ::Type, ::Type...) = WeakPreserving()
  end
end

const _STRONG_FUNCTIONS = (:*,)
for f in _STRONG_FUNCTIONS
  @eval begin
    ZeroPreserving(::typeof($f), ::AbstractArray, ::AbstractArray...) = StrongPreserving()
    ZeroPreserving(::typeof($f), ::Type, ::Type...) = StrongPreserving()
  end
end

# map(!)
# ------
@interface I::AbstractSparseArrayInterface function Base.map(
  f, A::AbstractArray, Bs::AbstractArray...
)
  T = Base.Broadcast.combine_eltypes(f, (A, Bs...))
  C = similar(I, T, size(A))
  return @interface I map!(f, C, A, Bs...)
end

@interface ::AbstractSparseArrayInterface function Base.map!(
  f, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...
)
  return _map!(f, ZeroPreserving(f, A, Bs...), C, A, Bs...)
end

function _map!(
  f, ::StrongPreserving, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...
)
  checkshape(C, A, Bs...)
  style = IndexStyle(C, A, Bs...)
  unaliased = map(Base.Fix1(Base.unalias, C), (A, Bs...))
  zero!(C)
  for I in intersect(eachstoredindex.(Ref(style), unaliased)...)
    @inbounds C[I] = f(ith_all(I, unaliased)...)
  end
  return C
end
function _map!(
  f, ::WeakPreserving, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...
)
  checkshape(C, A, Bs...)
  style = IndexStyle(C, A, Bs...)
  unaliased = map(Base.Fix1(Base.unalias, C), (A, Bs...))
  zero!(C)
  for I in union(eachstoredindex.(Ref(style), unaliased)...)
    @inbounds C[I] = f(ith_all(I, unaliased)...)
  end
  return C
end
function _map!(f, ::NonPreserving, C::AbstractArray, A::AbstractArray, Bs::AbstractArray...)
  checkshape(C, A, Bs...)
  unaliased = map(Base.Fix1(Base.unalias, C), (A, Bs...))
  for I in eachindex(C, A, Bs...)
    @inbounds C[I] = f(ith_all(I, unaliased)...)
  end
  return C
end

# Derived functions
# -----------------
@interface I::AbstractSparseArrayInterface Base.copyto!(
  C::AbstractArray, A::AbstractArray
) = @interface I map!(identity, C, A)

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
