
export CopyArray

_isdense(A::DenseArray) = true
_isdense(A::AbstractArray) = A===parent(A) ? false : _isdense(parent(A))

_copy(A) = _isdense(A) ? CopyArray(A) : copy(A)

# The idea is that @adjoint + will have Δ -> (_copy(Δ), _copy(Δ)),
# but neither actually makes a copy until it wishes to write into Δ, and then only once.

mutable struct CopyArray{T,N,A} <: AbstractArray{T,N}
  data::A
  write::Bool  # write=true means it's safe to write
  CopyArray(data::A, write::Bool=false) where {A<:AbstractArray{T,N}} where {T,N} = new{T,N,A}(data, write)
end

CopyArray(A::CopyArray) = A.write ? CopyArray(copy(A.data)) : CopyArray(A.data)
Base.reshape(A::CopyArray, sz::Tuple{Vararg{Int}}) = CopyArray(reshape(A.data, sz), A.write)

Base.size(A::CopyArray) = size(A.data)
Base.axes(A::CopyArray) = axes(A.data)
Base.parent(A::CopyArray) = A.data

Base.copy(A::CopyArray) = A.write ? A.data : copy(A.data)  # always returns an array safe to write into

Base.similar(A::CopyArray, ::Type{T}, dims::Tuple{Int, Vararg{Int}}) where {T} = similar(A.data, T, dims)

Base.@propagate_inbounds Base.getindex(A::CopyArray, i...) = getindex(A.data, i...)

Base.@propagate_inbounds function Base.setindex!(A::CopyArray, v, i...)
  if !A.write 
    @debug "copying for setindex" summary(A.data)
    A.data = copy(A.data)
    A.write = true
  end
  setindex!(A.data, v, i...)
end

function Base.showarg(io::IO, A::CopyArray, toplevel)
  print(io, "CopyArray(")
  Base.showarg(io, parent(A), false)
  if A.write
    print(io, ", ", A.write)
  end
  print(io, ')')
end

# Haven't tested for speed, but hoped to un-wrap whole arrays when possible,
# and perform copy check just once. Broadcasting is called by accum(),
# and reductions are used in unbroadcast.

Broadcast.broadcastable(A::CopyArray) = A.data  # unwraps on the right

function Base.copyto!(A::CopyArray, bc::Broadcast.Broadcasted) 
  if !A.write
    @debug "copying for broadcast" summary(A.data)
    A.data = copy(A.data)  # some bugs! e.g. copy(PermutedDimsArray(rand(3,3),(2,1))) isa Matrix
    A.write = true
  end
  copyto!(A.data, bc)  # returns the plain array, as the wrapper is no longer useful
end

_nowrite(A) = A
_nowrite(A::CopyArray) = parent(A)

# @less sum!(identity, rand(3), rand(3,3))

for op in [:add_sum, :mul_prod, :max, :min, :&, :|]
  @eval begin
    Base.initarray!(R::AbstractArray, f::typeof(Base.$op), init::Bool, A::CopyArray) = _initwrite(R, f, init, A)
    Base.initarray!(R::CopyArray, f::typeof(Base.$op), init::Bool, A::AbstractArray) = _initwrite(R, f, init, A)
    Base.initarray!(R::CopyArray, f::typeof(Base.$op), init::Bool, A::CopyArray) = _initwrite(R, f, init, A)
  end
end
_initwrite(R, f, init, A) = Base.initarray!(R, f, init, _nowrite(A))
function _initwrite(R::CopyArray, f, init, A)
  if !R.write
    @debug "copying for mapreduce" summary(R.data)
    R.data = copy(R.data)
    R.write = true
  end
  Base.initarray!(R.data, f, init, _nowrite(A))
end

#=

julia> ENV["JULIA_DEBUG"] = "all"
"all"

julia> let rr = ones(3)
       xx = CopyArray(rr); yy = CopyArray(rr);
       sum!(xx, yy .* rr')
       sum!(xx, 100 .* yy .* rr')
       end
┌ Debug: copying for mapreduce
│   summary(R.data) = "3-element Vector{Float64}"
└ @ Zygote ~/.julia/dev/Zygote/src/lib/copywrite.jl:93
3-element Vector{Float64}:
 300.0
 300.0
 300.0

=#

using LinearAlgebra

# We need to remove the wrapper if we hope for * to go to BLAS, but dispatch is pretty messy.

CopyVector{T} = CopyArray{T,1}
CopyMatrix{T} = CopyArray{T,2}
CopyVecOrMat{T} = Union{CopyVector{T}, CopyMatrix{T}}

LinearAlgebra.generic_matvecmul!(C::AbstractVector, tA, A::CopyVecOrMat, B::AbstractVector, _add::LinearAlgebra.MulAddMul) = _mulv(C, tA, A, B, _add)
LinearAlgebra.generic_matvecmul!(C::AbstractVector, tA, A::AbstractVecOrMat, B::CopyVector, _add::LinearAlgebra.MulAddMul) = _mulv(C, tA, A, B, _add)
LinearAlgebra.generic_matvecmul!(C::AbstractVector, tA, A::CopyVecOrMat, B::CopyVector, _add::LinearAlgebra.MulAddMul) = _mulv(C, tA, A, B, _add)
function _mulv(C, tA, A0, B0, _add)
  A, B = _nowrite(A0), _nowrite(B0)
  if tA == 'N'
    mul!(C, A, B, _add.alpha, _add.beta)
  elseif tA == 'T'
    mul!(C, transpose(A), B, _add.alpha, _add.beta)
  elseif tA == 'C'
    mul!(C, A', B, _add.alpha, _add.beta)
  end
  C
end

# LinearAlgebra.generic_matmatmul!(C::AbstractMatrix, tA, tB, A::CopyMatrix, B::AbstractMatrix, _add::LinearAlgebra.MulAddMul) 
# LinearAlgebra.generic_matmatmul!(C::AbstractMatrix, tA, tB, A::AbstractMatrix, B::CopyMatrix, _add::LinearAlgebra.MulAddMul) 
# LinearAlgebra.generic_matmatmul!(C::AbstractMatrix, tA, tB, A::CopyMatrix, B::CopyMatrix, _add::LinearAlgebra.MulAddMul) 

using ChainRules

# Maybe better unwrap earlier. Ordinary rules never mutate their inputs, so it's always safe to unwrap?
# No, a rule could return the array unchanged, and this would free it to be mutated, affecting its cousins.

# @inline function wrap_chainrules_input(x::CopyArray)
#   @debug "unwrapping for chainrules" summary(x.data)
#   parent(x)
# end

_pointer(A) = NaN  # compares != self, although perhaps we're guaranteed that initial ptr isa Ptr
_pointer(A::DenseArray) = pointer(A)  # pointer survives reshape, objectid does not
_pointer(A::AbstractArray) = A===parent(A) ? false : _pointer(parent(A))

@inline function (s::ZBack)(dy::CopyArray)
  ptr = _pointer(dy.data)
  @debug "unwrapping for chainrules" summary(dy.data) ptr s.back
  dxs = wrap_chainrules_output(s.back(wrap_chainrules_input(dy.data)))
  map(dxs) do dx
    if _pointer(dx) == ptr
      @debug "re-wrapping for chainrules" summary(dy.data) ptr
      CopyArray(dx)
    else
      dx
    end
  end
end

# It would be even better if, after the rule has run, you could mark the array as free
# with a shared counter, so that all other "copies" know, and you can safely 
# mutate the last man standing. Decrement on unwrap, increment on wrap. Maybe threads complicate that?
# For example, this copy could be avoided:

#=

julia> ENV["JULIA_DEBUG"] = "all";

julia> gradient(x -> sum(abs, (x * x) + x), [1 2; 3 4])[1]
┌ Debug: unwrapping for chainrules
│   summary(dy.data) = "2×2 Matrix{Int64}"
│   ptr = Ptr{Int64} @0x00000002822f7a80
│   s.back = (::ChainRules.var"#times_pullback#1527"{Matrix{Int64}, Matrix{Int64}}) (generic function with 1 method)
└ @ Zygote ~/.julia/dev/Zygote/src/lib/copywrite.jl:151
┌ Debug: copying for broadcast
│   summary(A.data) = "2×2 Matrix{Int64}"
└ @ Zygote ~/.julia/dev/Zygote/src/lib/copywrite.jl:58
2×2 Matrix{Int64}:
  8  12
 10  14

=#

