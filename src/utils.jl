function qrm_refine(spmat :: qrm_spmat{T}, spfct :: qrm_spfct{T}, x :: AbstractVector{T}, z :: AbstractVector{T}) where T
  Δx = similar(x)
  y = similar(x, spfct.fct.m)
  x_refined = copy(x)

  qrm_refine!(spmat, spfct, x_refined, z, Δx, y)
  return x_refined
end

function qrm_refine!(spmat :: qrm_spmat{T}, spfct :: qrm_spfct{T}, x :: AbstractVector{T}, z :: AbstractVector{T}, Δx :: AbstractVector{T}, y :: AbstractVector{T}) where T
  @assert length(x) == spfct.fct.n
  @assert length(z) == spfct.fct.n
  @assert length(Δx) == spfct.fct.n
  @assert length(y) == spfct.fct.m

  transp = T <: Real ? 't' : 'c'

  qrm_spmat_mv!(spmat, T(1), x, T(0), y, transp = 'n')
  qrm_spmat_mv!(spmat, T(1), y, T(0), Δx, transp = transp)
  @. Δx = z - Δx 

  qrm_solve!(spfct, Δx, y, transp = transp)
  qrm_solve!(spfct, y, Δx, transp = 'n')
  @. x = x + Δx
end

#Given an underdetermined, rank defficient system Ax = b, compute the least-norm solution with Golub-Riley iteration
# TODO: add something for rank defficient least squares as well (i.e compute (Aᵀ)†z)
function qrm_golub_riley!(
  spmat :: qrm_spmat{T},  
  xₖ :: AbstractVector{T}, 
  Δx ::AbstractVector{T},
  bₖ :: AbstractVector{T},
  b :: AbstractVector{T};  
  α :: T = eps(T)^(1//4), 
  max_iter ::Int = 10, tol ::T = T(1e-3)
  ) where {T <: Real}
  
  m = spmat.mat.m
  nz = spmat.mat.nz
  n = spmat.mat.n - spmat.mat.m # The matrix A is (m,n), spmat represents a (m,n+m) matrix

  @assert n > 0
  @assert all(spmat.irn[i + nz - m] == Cint(i) for i = 1:m)
  @assert all(spmat.jcn[i + nz - m] == Cint(i+n) for i = 1:m)
  spmat.val[nz - m + 1:nz] .= α
  @assert length(b) == spmat.mat.m
  @assert length(xₖ) == spmat.mat.n
  @assert length(bₖ) == spmat.mat.m
  @assert length(Δx) == spmat.mat.n
 
  spfct = qrm_spfct_init(spmat)
  qrm_set(spfct, "qrm_keeph", 0)
  qrm_analyse!(spmat, spfct, transp ='t')
  qrm_factorize!(spmat, spfct, transp = 't')

  k = 0
  solved = false
  xₖ .= T(0)
  while k < max_iter && !solved

    qrm_spmat_mv!(spmat, T(1), xₖ, T(0), bₖ)
    @. bₖ = b - bₖ

    qrm_solve!(spfct, bₖ, Δx, transp = 't')
    qrm_solve!(spfct, Δx, bₖ, transp = 'n')

    qrm_spmat_mv!(spmat, T(1), bₖ, T(0), Δx, transp = 't')

    @. xₖ = xₖ + Δx
    
    solved = norm(Δx) ≤ tol*norm(xₖ)
    println(norm(bₖ))
    k = k + 1
  end
end