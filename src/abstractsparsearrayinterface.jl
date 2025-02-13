using DerivableInterfaces: DerivableInterfaces, @derive, @interface, AbstractArrayInterface

# This is to bring `ArrayLayouts.zero!` into the namespace
# since it is considered part of the sparse array interface.
using ArrayLayouts: zero!
using Random: Random, AbstractRNG, default_rng

function eachstoredindex end
function getstoredindex end
function getunstoredindex end
function isstored end
function setstoredindex! end
function setunstoredindex! end
function storedlength end
function storedpairs end
function storedvalues end

# Replace the function for accessing
# unstored values.
function set_getunstoredindex end

# Generic functionality for converting to a
# dense array, trying to preserve information
# about the array (such as which device it is on).
# TODO: Maybe call `densecopy`?
# TODO: Make sure this actually preserves the device,
# maybe use `TypeParameterAccessors.unwrap_array_type`.
# TODO: Turn into an `@interface` function.
function densearray(a::AbstractArray)
  # TODO: `set_ndims(unwrap_array_type(a), ndims(a))(a)`
  # Maybe define `densetype(a) = set_ndims(unwrap_array_type(a), ndims(a))`.
  # Or could use `unspecify_parameters(unwrap_array_type(a))(a)`.
  return Array(a)
end

@doc """
    spzeros([T::Type], dims) -> A::SparseArrayDOK{T}

Create an empty size `dims` sparse array.
The optional `T` argument specifies the element type, which defaults to `Float64`.
""" spzeros

spzeros(dims::Dims) = spzeros(Float64, dims)
spzeros(::Type{T}, dims::Dims) where {T} = SparseArrayDOK{T}(undef, dims)

@doc """
    sprand([rng], [T::Type], dims; density::Real=0.5, rfn::Function=rand) -> A::SparseArrayDOK{T}

Create a random size `dims` sparse array in which the probability of any element being stored is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `T` argument specifies the element type, which defaults to `Float64`.
The optional `rfn` argument can be used to control the type of random elements.

See also [`sprand!`](@ref).
""" sprand

function sprand(::Type{T}, dims::Dims; kwargs...) where {T}
  return sprand(default_rng(), T, dims; kwargs...)
end
sprand(dims::Dims; kwargs...) = sprand(default_rng(), Float64, dims; kwargs...)
function sprand(rng::AbstractRNG, dims::Dims; kwargs...)
  return sprand(rng, Float64, dims; kwargs...)
end
function sprand(rng::AbstractRNG, ::Type{T}, dims::Dims; kwargs...) where {T}
  A = SparseArrayDOK{T}(undef, dims)
  sprand!(rng, A; kwargs...)
  return A
end

@doc """
    sprand!([rng], A::AbstractArray; density::Real=0.5, rfn::Function=rand) -> A

Overwrite part of an array with random entries, where the probability of overwriting is independently given by `density`.
The optional `rng` argument specifies a random number generator, see also `Random`.
The optional `rfn` argument can be used to control the type of random elements.

See also [`sprand`](@ref).
""" sprand!

sprand!(A::AbstractArray; kwargs...) = sprand!(default_rng(), A; kwargs...)
function sprand!(
  rng::AbstractRNG, A::AbstractArray; density::Real=0.5, rfn::Function=Random.rand
)
  ArrayLayouts.zero!(A)
  rand_inds = Random.randsubseq(rng, eachindex(A), density)
  rand_entries = rfn(rng, eltype(A), length(rand_inds))
  for (I, v) in zip(rand_inds, rand_entries)
    A[I] = v
  end
end

# Minimal interface for `SparseArrayInterface`.
# Fallbacks for dense/non-sparse arrays.
@interface ::AbstractArrayInterface isstored(a::AbstractArray, I::Int...) = true
@interface ::AbstractArrayInterface eachstoredindex(a::AbstractArray) = eachindex(a)
@interface ::AbstractArrayInterface getstoredindex(a::AbstractArray, I::Int...) =
  getindex(a, I...)
@interface ::AbstractArrayInterface function setstoredindex!(
  a::AbstractArray, value, I::Int...
)
  setindex!(a, value, I...)
  return a
end
# TODO: Should this error by default if the value at the index
# is stored? It could be disabled with something analogous
# to `checkbounds`, like `checkstored`/`checkunstored`.
@interface ::AbstractArrayInterface function setunstoredindex!(
  a::AbstractArray, value, I::Int...
)
  # TODO: Make this a `MethodError`?
  return error("Not implemented.")
end

# TODO: Use `Base.to_indices`?
isstored(a::AbstractArray, I::CartesianIndex) = isstored(a, Tuple(I)...)
# TODO: Use `Base.to_indices`?
getstoredindex(a::AbstractArray, I::CartesianIndex) = getstoredindex(a, Tuple(I)...)
# TODO: Use `Base.to_indices`?
getunstoredindex(a::AbstractArray, I::CartesianIndex) = getunstoredindex(a, Tuple(I)...)
# TODO: Use `Base.to_indices`?
function setstoredindex!(a::AbstractArray, value, I::CartesianIndex)
  return setstoredindex!(a, value, Tuple(I)...)
end
# TODO: Use `Base.to_indices`?
function setunstoredindex!(a::AbstractArray, value, I::CartesianIndex)
  return setunstoredindex!(a, value, Tuple(I)...)
end

# Interface defaults.
# TODO: Have a fallback that handles element types
# that don't define `zero(::Type)`.
@interface ::AbstractArrayInterface getunstoredindex(a::AbstractArray, I::Int...) =
  zero(eltype(a))

# DerivableInterfacesd interface.
@interface ::AbstractArrayInterface storedlength(a::AbstractArray) = length(storedvalues(a))
@interface ::AbstractArrayInterface storedpairs(a::AbstractArray) =
  map(I -> I => getstoredindex(a, I), eachstoredindex(a))

@interface ::AbstractArrayInterface function eachstoredindex(as::AbstractArray...)
  return eachindex(as...)
end

@interface ::AbstractArrayInterface storedvalues(a::AbstractArray) = a

# Automatically derive the interface for all `AbstractArray` subtypes.
# TODO: Define `SparseArrayInterfaceOps` derivable trait and rewrite this
# as `@derive AbstractArray SparseArrayInterfaceOps`.
@derive (T=AbstractArray,) begin
  SparseArraysBase.eachstoredindex(::T)
  SparseArraysBase.eachstoredindex(::T...)
  SparseArraysBase.getstoredindex(::T, ::Int...)
  SparseArraysBase.getunstoredindex(::T, ::Int...)
  SparseArraysBase.isstored(::T, ::Int...)
  SparseArraysBase.setstoredindex!(::T, ::Any, ::Int...)
  SparseArraysBase.setunstoredindex!(::T, ::Any, ::Int...)
  SparseArraysBase.storedlength(::T)
  SparseArraysBase.storedpairs(::T)
  SparseArraysBase.storedvalues(::T)
end

# TODO: Add `ndims` type parameter, like `Base.Broadcast.AbstractArrayStyle`.
# TODO: This isn't used to define interface functions right now.
# Currently, `@interface` expects an instance, probably it should take a
# type instead so fallback functions can use abstract types.
abstract type AbstractSparseArrayInterface <: AbstractArrayInterface end

function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractSparseArrayInterface, interface2::AbstractSparseArrayInterface
)
  return error("Rule not defined.")
end
function DerivableInterfaces.combine_interface_rule(
  interface1::Interface, interface2::Interface
) where {Interface<:AbstractSparseArrayInterface}
  return interface1
end
function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractSparseArrayInterface, interface2::AbstractArrayInterface
)
  return interface1
end
function DerivableInterfaces.combine_interface_rule(
  interface1::AbstractArrayInterface, interface2::AbstractSparseArrayInterface
)
  return interface2
end

to_vec(x) = vec(collect(x))
to_vec(x::AbstractArray) = vec(x)

# A view of the stored values of an array.
# Similar to: `@view a[collect(eachstoredindex(a))]`, but the issue
# with that is it returns a `SubArray` wrapping a sparse array, which
# is then interpreted as a sparse array so it can lead to recursion.
# Also, that involves extra logic for determining if the indices are
# stored or not, but we know the indices are stored so we can use
# `getstoredindex` and `setstoredindex!`.
# Most sparse arrays should overload `storedvalues` directly
# and avoid this wrapper since it adds extra indirection to
# access stored values.
struct StoredValues{T,A<:AbstractArray{T},I} <: AbstractVector{T}
  array::A
  storedindices::I
end
StoredValues(a::AbstractArray) = StoredValues(a, to_vec(eachstoredindex(a)))
Base.size(a::StoredValues) = size(a.storedindices)
Base.getindex(a::StoredValues, I::Int) = getstoredindex(a.array, a.storedindices[I])
function Base.setindex!(a::StoredValues, value, I::Int)
  return setstoredindex!(a.array, value, a.storedindices[I])
end

@interface ::AbstractSparseArrayInterface storedvalues(a::AbstractArray) = StoredValues(a)

@interface ::AbstractSparseArrayInterface function eachstoredindex(
  a1::AbstractArray, a2::AbstractArray, a_rest::AbstractArray...
)
  # TODO: Make this more customizable, say with a function
  # `combine/promote_storedindices(a1, a2)`.
  return union(eachstoredindex.((a1, a2, a_rest...))...)
end

@interface ::AbstractSparseArrayInterface function eachstoredindex(a::AbstractArray)
  # TODO: Use `MethodError`?
  return error("Not implemented.")
end

# We restrict to `I::Vararg{Int,N}` to allow more general functions to handle trailing
# indices and linear indices.
@interface ::AbstractSparseArrayInterface function Base.getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  !isstored(a, I...) && return getunstoredindex(a, I...)
  return getstoredindex(a, I...)
end

# We restrict to `I::Vararg{Int,N}` to allow more general functions to handle trailing
# indices and linear indices.
@interface ::AbstractSparseArrayInterface function Base.setindex!(
  a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  if !isstored(a, I...)
    # Don't set the value if it is zero, but only check
    # if it is zero if the elements are numbers since otherwise
    # it may be nontrivial to check.
    eltype(a) <: Number && iszero(value) && return a
    setunstoredindex!(a, value, I...)
    return a
  end
  setstoredindex!(a, value, I...)
  return a
end

# TODO: This may need to be defined in `sparsearraydok.jl`, after `SparseArrayDOK`
# is defined. And/or define `default_type(::SparseArrayStyle, T::Type) = SparseArrayDOK{T}`.
@interface ::AbstractSparseArrayInterface function Base.similar(
  a::AbstractArray, T::Type, size::Tuple{Vararg{Int}}
)
  # TODO: Define `default_similartype` or something like that?
  return SparseArrayDOK{T}(undef, size)
end

# map over a specified subset of indices of the inputs.
function map_indices! end

@interface interface::AbstractArrayInterface function map_indices!(
  indices, f, a_dest::AbstractArray, as::AbstractArray...
)
  for I in indices
    a_dest[I] = f(map(a -> a[I], as)...)
  end
  return a_dest
end

# Only map the stored values of the inputs.
function map_stored! end

@interface interface::AbstractArrayInterface function map_stored!(
  f, a_dest::AbstractArray, as::AbstractArray...
)
  @interface interface map_indices!(eachstoredindex(as...), f, a_dest, as...)
  return a_dest
end

# Only map all values, not just the stored ones.
function map_all! end

@interface interface::AbstractArrayInterface function map_all!(
  f, a_dest::AbstractArray, as::AbstractArray...
)
  @interface interface map_indices!(eachindex(as...), f, a_dest, as...)
  return a_dest
end

using ArrayLayouts: ArrayLayouts, zero!

# `zero!` isn't defined in `Base`, but it is defined in `ArrayLayouts`
# and is useful for sparse array logic, since it can be used to empty
# the sparse array storage.
# We use a single function definition to minimize method ambiguities.
@interface interface::AbstractSparseArrayInterface function ArrayLayouts.zero!(
  a::AbstractArray
)
  # More generally, this codepath could be taking if `zero(eltype(a))`
  # is defined and the elements are immutable.
  f = eltype(a) <: Number ? Returns(zero(eltype(a))) : zero!
  return @interface interface map_stored!(f, a, a)
end

# Determines if a function preserves the stored values
# of the destination sparse array.
# The current code may be inefficient since it actually
# accesses an unstored element, which in the case of a
# sparse array of arrays can allocate an array.
# Sparse arrays could be expected to define a cheap
# unstored element allocator, for example
# `get_prototypical_unstored(a::AbstractArray)`.
function preserves_unstored(f, a_dest::AbstractArray, as::AbstractArray...)
  I = first(eachindex(as...))
  return iszero(f(map(a -> getunstoredindex(a, I), as)...))
end

@interface interface::AbstractSparseArrayInterface function Base.map!(
  f, a_dest::AbstractArray, as::AbstractArray...
)
  indices = if !preserves_unstored(f, a_dest, as...)
    eachindex(a_dest)
  elseif any(a -> a_dest !== a, as)
    as = map(a -> Base.unalias(a_dest, a), as)
    @interface interface zero!(a_dest)
    eachstoredindex(as...)
  else
    eachstoredindex(a_dest)
  end
  @interface interface map_indices!(indices, f, a_dest, as...)
  return a_dest
end

# `f::typeof(norm)`, `op::typeof(max)` used by `norm`.
function reduce_init(f, op, as...)
  # TODO: Generalize this.
  @assert isone(length(as))
  a = only(as)
  ## TODO: Make this more efficient for block sparse
  ## arrays, in that case it allocates a block. Maybe
  ## it can use `FillArrays.Zeros`.
  return f(getunstoredindex(a, first(eachindex(a))))
end

@interface ::AbstractSparseArrayInterface function Base.mapreduce(
  f, op, as::AbstractArray...; init=reduce_init(f, op, as...), kwargs...
)
  # TODO: Generalize this.
  @assert isone(length(as))
  a = only(as)
  output = mapreduce(f, op, storedvalues(a); init, kwargs...)
  ## TODO: Bring this check back, or make the function more general.
  ## f_notstored = apply_notstored(f, a)
  ## @assert isequal(op(output, eltype(output)(f_notstored)), output)
  return output
end

abstract type AbstractSparseArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

@derive AbstractSparseArrayStyle AbstractArrayStyleOps

struct SparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end

SparseArrayStyle{M}(::Val{N}) where {M,N} = SparseArrayStyle{N}()

@interface ::AbstractSparseArrayInterface function Broadcast.BroadcastStyle(type::Type)
  return SparseArrayStyle{ndims(type)}()
end

using ArrayLayouts: ArrayLayouts, MatMulMatAdd

abstract type AbstractSparseLayout <: ArrayLayouts.MemoryLayout end

function ArrayLayouts.sub_materialize(::AbstractSparseLayout, a::AbstractArray, axes::Tuple)
  a_dest = similar(a)
  a_dest .= a
  return a_dest
end

function mul_indices(I1::CartesianIndex{2}, I2::CartesianIndex{2})
  if I1[2] ≠ I2[1]
    return nothing
  end
  return CartesianIndex(I1[1], I2[2])
end

using LinearAlgebra: mul!
function default_mul!!(
  a_dest::AbstractMatrix,
  a1::AbstractMatrix,
  a2::AbstractMatrix,
  α::Number=true,
  β::Number=false,
)
  mul!(a_dest, a1, a2, α, β)
  return a_dest
end

function default_mul!!(
  a_dest::Number, a1::Number, a2::Number, α::Number=true, β::Number=false
)
  return a1 * a2 * α + a_dest * β
end

# a1 * a2 * α + a_dest * β
function sparse_mul!(
  a_dest::AbstractArray,
  a1::AbstractArray,
  a2::AbstractArray,
  α::Number=true,
  β::Number=false;
  (mul!!)=(default_mul!!),
)
  a_dest .*= β
  β′ = one(Bool)
  for I1 in eachstoredindex(a1)
    for I2 in eachstoredindex(a2)
      I_dest = mul_indices(I1, I2)
      if !isnothing(I_dest)
        a_dest[I_dest] = mul!!(a_dest[I_dest], a1[I1], a2[I2], α, β′)
      end
    end
  end
  return a_dest
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{<:AbstractSparseLayout,<:AbstractSparseLayout,<:AbstractSparseLayout}
)
  sparse_mul!(m.C, m.A, m.B, m.α, m.β)
  return m.C
end

struct SparseLayout <: AbstractSparseLayout end

@interface ::AbstractSparseArrayInterface function ArrayLayouts.MemoryLayout(type::Type)
  return SparseLayout()
end

# Like `Char` but prints without quotes.
struct UnquotedChar <: AbstractChar
  char::Char
end
Base.show(io::IO, c::UnquotedChar) = print(io, c.char)
Base.show(io::IO, ::MIME"text/plain", c::UnquotedChar) = show(io, c)

function getunstoredindex_show(a::AbstractArray, I::Int...)
  return UnquotedChar('⋅')
end

@interface ::AbstractSparseArrayInterface function Base.show(
  io::IO, mime::MIME"text/plain", a::AbstractArray
)
  summary(io, a)
  isempty(a) && return nothing
  print(io, ":")
  println(io)
  a′ = ReplacedUnstoredSparseArray(a, getunstoredindex_show)
  Base.print_array(io, a′)
  return nothing
end
