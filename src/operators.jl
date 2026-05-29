# SBP-SAT operators for a single element: their type, construction, and the
# SAT increment primitive (`_sat_increment`) used by both the 1D and 3D
# kernels.

# Operator container with N as a compile-time parameter. The H/Hinv slot
# is parameterised on `Hmat` so the GLL branch can store an `SDiagonal`
# (truly diagonal — `H` is the GLL quadrature weights) while the Rational
# branch keeps a full `SMatrix` (the Vandermonde-based H is dense).
# Specialised `_sat_increment` methods dispatch on the Hmat type to skip
# the trivially-zero off-diagonal multiplications in the GLL case.
struct SBPOps{N, T, NN, Hmat<:AbstractMatrix{T}}
    B       :: SMatrix{N, N, T, NN}
    G       :: SMatrix{N, N, T, NN}
    H       :: Hmat
    Hinv    :: Hmat
    HinvG_L :: SVector{N, T}
    HinvG_R :: SVector{N, T}
    D       :: SMatrix{N, N, T, NN}
    L       :: SMatrix{N, N, T, NN}
end

################################################################################

function make_element(::Type{T}, N::Int) where {T}
    ns = 0:N-1
    x0 = T(0//1)
    x1 = T(1//1)

    if T <: Rational
        # Equispaced "vertex-centred" points keep the operator construction
        # exact in rational arithmetic.
        h = (x1-x0) / (N-1)
        xs = x0 .+ h * ns
    else
        # Gauss-Lobatto-Legendre collocation points (better-conditioned SBP
        # operators than equispaced). `gausslobatto` returns nodes on [-1,1];
        # map them linearly to [x0, x1].
        ξ, _ = gausslobatto(N)
        xs = T.((ξ .+ 1) ./ 2) .* (x1 - x0) .+ x0
        # GLL points cluster near the endpoints; the minimum spacing sets
        # the effective CFL length scale.
        h = minimum(diff(xs))
    end

    return (;N,ns,x0,x1,h,xs)
end

function make_domain(::Type{T}, N::Int, x0, x1) where {T}
    ns = 0:N-1

    x0 = T(x0)
    x1 = T(x1)
    # N elements, "cell-centred"
    h = (x1-x0) / N
    xs = x0 .+ h * ns

    return (;N,ns,x0,x1,h,xs)
end

# Top-level entry: dispatch to a `Val{N}`-parameterised implementation so
# the returned `SBPOps` has fully-concrete static types. The dispatch site
# is type-unstable but `make_operators` is called once per simulation, so
# the cost is irrelevant.
make_operators(dom) = _make_operators(Val(dom.N), dom)

function _make_operators(::Val{N}, dom) where {N}
    _, ns, x0, x1, h, xs = dom
    T = eltype(xs)

    # Boundary operator (same in both branches)
    B = Diagonal([ifelse(n==ns[begin], -T(1), ifelse(n==ns[end], +T(1), zero(T)))
                  for n in ns])

    if T <: Rational
        # Polynomial-exactness construction. Exact in rational arithmetic.
        # Numerically catastrophic in floating point for N ≳ 13 because the
        # Vandermonde-style `lhs` matrix has condition number ∼ 4ᴺ.
        G = let
            lhs = [x^p for x in xs, p in 0:N-1]
            rhs = [p==0 ? 0 : p * x^(p-1) for x in xs, p in 0:N-1]
            G = rhs / lhs
            @assert G * lhs == rhs
            @assert all(G * xs.^0 .== 0)
            @assert all(G * xs.^p == p * xs.^(p-1) for p in 1:N-1)
            G
        end
        H = let
            u = [x^p for x in xs, p in 0:N-1]
            v = [x^q for x in xs, q in 0:N-1]
            r = [1//(p+q+1) * (x1^(p+q+1) - x0^(p+q+1)) for p in 0:N-1, q in 0:N-1]
            H = (u' \ r) / v
            @assert issymmetric(H)
            @assert all(>(0), eigvals(H))
            @assert all(u[:,k]' * H * v[:,l] == r[k,l] for k in 1:N, l in 1:N)
            H
        end
        D = H \ (B - (H*G)')
        @assert D == G
        @assert all(D * xs.^0 .== 0)
        @assert all(D * xs.^p == p * xs.^(p-1) for p in 1:N-1)
    else
        # GLL spectral-collocation operators via PolynomialBases.jl —
        # numerically stable for any N. The basis is defined on [−1, 1];
        # we map to [x0, x1] via the Jacobian dx/dξ = (x1−x0)/2.
        basis = LobattoLegendre(N - 1, T)
        jac   = (x1 - x0) / T(2)
        G = basis.D ./ jac                # ∂/∂x = (1/jac) · ∂/∂ξ
        H = Diagonal(basis.weights .* jac) # ∫·dx = jac · ∫·dξ
        D = G                              # diagonal-norm GLL collocation: D = G
    end

    # Laplacian (without boundary conditions)
    L = D * G

    Hinv = inv(H)

    # Precompute Hinv · (first/last rows of G), used by the SAT consistency
    # term `Hinv · Gᵀ · B · …` which only touches the first and last columns
    # of `Gᵀ` (i.e., the first and last rows of `G`).
    HinvG_L = Hinv * G[1, :]
    HinvG_R = Hinv * G[N, :]

    # Pick a static container for `H`/`Hinv`. The GLL branch produces a
    # genuinely diagonal `H` (the quadrature weights) — store as `SDiagonal`
    # so the SAT inner loop skips the off-diagonal zeros. The Rational
    # (Vandermonde) branch produces a dense H — store as `SMatrix`.
    H_static, Hinv_static = if H isa Diagonal
        SDiagonal(SVector{N, T}(H.diag)), SDiagonal(SVector{N, T}(Hinv.diag))
    else
        SMatrix{N, N, T}(H), SMatrix{N, N, T}(Hinv)
    end

    return SBPOps(
        SMatrix{N, N, T}(B),
        SMatrix{N, N, T}(G),
        H_static,
        Hinv_static,
        SVector{N, T}(HinvG_L),
        SVector{N, T}(HinvG_R),
        SMatrix{N, N, T}(D),
        SMatrix{N, N, T}(L),
    )
end

################################################################################

# SIPG-style SAT penalty. Exchanges both u and Gu across each face.
#
#     L_SAT u = L u + H⁻¹ [ (1/2) Gᵀ B Δu  −  (1/2) B ΔGu  −  τ |B| Δu ]
#
# Per-face consistency weight: 1 at outer Dirichlet faces (full Nitsche),
# 1/2 at interior interfaces (SIPG averages between two elements).
#
# Two specialisations dispatched on the type of `ops.Hinv`:
#
#   * Dense `SMatrix` (Rational branch): full `b · Hinv[:, k]` SVector ops.
#   * `SDiagonal`     (GLL branch):       only the diagonal entry contributes,
#                                         so we patch just the first/last
#                                         component of the base increment.
@inline function _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR,
                                ops::SBPOps{N,T,NN,<:SMatrix}, τ) where {N, T, NN}
    cL = -αL * ΔuL
    cR =  αR * ΔuR
    b1 =  T(1//2) * ΔGuL - τ * ΔuL
    bN = -T(1//2) * ΔGuR - τ * ΔuR
    return cL * ops.HinvG_L + cR * ops.HinvG_R +
           b1 * ops.Hinv[:, 1] + bN * ops.Hinv[:, N]
end

@inline function _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR,
                                ops::SBPOps{N,T,NN,<:Diagonal}, τ) where {N, T, NN}
    cL = -αL * ΔuL
    cR =  αR * ΔuR
    b1 =  T(1//2) * ΔGuL - τ * ΔuL
    bN = -T(1//2) * ΔGuR - τ * ΔuR
    inc = cL * ops.HinvG_L + cR * ops.HinvG_R
    # Hinv is diagonal: `Hinv[:, 1]` and `Hinv[:, N]` are zero except at the
    # endpoints. Patch only those two entries to avoid 2·(N−1) zero mul-adds.
    inc = setindex(inc, inc[1] + b1 * ops.Hinv[1, 1], 1)
    inc = setindex(inc, inc[N] + bN * ops.Hinv[N, N], N)
    return inc
end
