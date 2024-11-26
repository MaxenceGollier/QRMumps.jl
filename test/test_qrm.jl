Random.seed!(1234)
m = 200
n = 100
p = 5


@testset "Auxiliary functions" begin
  for str ∈ QRMumps.GICNTL ∪ QRMumps.PICNTL ∪ QRMumps.RCNTL
    qrm_get(str)
    qrm_set(str, 1)
  end

  @testset "precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    transp = (T <: Real) ? 't' : 'c'
    A = sprand(T, n, n, 0.3)
    spmat = qrm_spmat_init(A)
    spfct = qrm_analyse(spmat)
    qrm_factorize!(spmat, spfct)

    for str ∈ QRMumps.PICNTL ∪ QRMumps.RCNTL
      qrm_get(spfct, str)
      qrm_set(spfct, str, 1)
    end

    for str ∈ QRMumps.STATS
      qrm_get(spfct, str)
    end

    A = 2 * A
    qrm_update!(spmat, A)

    for ntype ∈ ('i', '1', 'f')
      qrm_spmat_nrm(spmat, ntype=ntype)
    end

    for ntype ∈ ('i', '1', '2')
      x = rand(T, 10)
      qrm_vecnrm(x, ntype=ntype)

      X = rand(T, 10, 5)
      nrm = qrm_vecnrm(X, ntype=ntype)

      X = 2 * X
      qrm_vecnrm!(X, nrm, ntype=ntype)
    end

    b = rand(T, n)
    x = rand(T, n)
    r = b - A * x
    qrm_residual_orth(spmat, r)
    qrm_residual_orth(spmat, r, transp=transp)
    qrm_residual_norm(spmat, b, x)
    qrm_residual_norm(spmat, b, x, transp=transp)

    B = rand(T, n, 5)
    X = rand(T, n, 5)
    R = B - A * X
    nrm = qrm_residual_orth(spmat, R)
    qrm_residual_orth!(spmat, R, nrm)
    qrm_residual_orth(spmat, R, transp=transp)
    nrm = qrm_residual_norm(spmat, B, X)
    qrm_residual_norm!(spmat, B, X, nrm)
    qrm_residual_norm(spmat, B, X, transp=transp)

    y = rand(T, n)
    y2 = copy(y)
    x = rand(T, n)
    Y = rand(T, n, 5)
    Y2 = copy(Y)
    X = rand(T, n, 5)
    α = rand(T)
    β = rand(T)

    mul!(y, spmat, x, α, β)
    mul!(y2, A, x, α, β)
    @test y ≈ y2
    mul!(Y, spmat, X, α, β)
    mul!(Y2, A, X, α, β)
    @test Y ≈ Y2

    mul!(y, Transpose(spmat), x, α, β)
    mul!(y2, Transpose(A), x, α, β)
    @test y ≈ y2
    mul!(Y, Transpose(spmat), X, α, β)
    mul!(Y2, Transpose(A), X, α, β)
    @test Y ≈ Y2

    mul!(y, Adjoint(spmat), x, α, β)
    mul!(y2, Adjoint(A), x, α, β)
    @test y ≈ y2
    mul!(Y, Adjoint(spmat), X, α, β)
    mul!(Y2, Adjoint(A), X, α, β)
    @test Y ≈ Y2
    
    A = sprand(T, m, n, 0.3)
    b = rand(T, n)
    spmat = qrm_spmat_init(A)
    spfct = qrm_analyse(spmat)
    qrm_set(spfct, "qrm_keeph", 0)
    qrm_factorize!(spmat, spfct)
    x₁ = qrm_solve(spfct,b, transp = transp)
    x = qrm_solve(spfct, x₁, transp = 'n')
    x_refined = qrm_refine(spmat, spfct, x, b)
    @test norm(b - A'*(A*x)) ≥ norm(b - A'*(A*x_refined))

    tol = (real(T) == Float32) ? 1e-3 : 1e-12
    qrm_init()
    A = sprand(T, n, m, 0.3)
    b = rand(T, n)

    spmat = qrm_spmat_init(A)
    spfct = qrm_spfct_init(spmat)
    qrm_set(spfct, "qrm_keeph", 0)
    x = zeros(T, m)
    Δx = similar(x)
    y = zeros(T, n)
    qrm_semi_normal!(spmat, spfct, x, b, Δx, y)

    x2 = qrm_min_norm(spmat, b)

    @test norm(A*x - b) ≤ tol
    @test abs(norm(x) - norm(x2)) ≤ tol

    A = sprand(T, m, n, 0.3)
    spmat = qrm_spmat_init(A)
    spfct = qrm_spfct_init(spmat)
    qrm_set(spfct, "qrm_keeph", 0)
    b = rand(T, m)
    x = zeros(T, n)
    Δx = similar(x)
    y = zeros(T, m)
    qrm_semi_normal!(spmat, spfct, x, b, Δx, y, type = "least_squares")

    x2 = qrm_least_squares(spmat, b)
    @test abs(norm(x) - norm(x2)) ≤ tol
  end
end
