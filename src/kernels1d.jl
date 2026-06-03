# 1D SBP-SAT discrete Laplacian: per-element kernel, global kernel, and the
# diagnostic global-Laplacian assembler used by tests. The two `apply_laplacian!`
# methods overload on shape:
#
#   * Vector / Vector — per-element kernel; the caller supplies the neighbour
#                       traces (`gL, gR, gGL, gGR`) and the per-face Nitsche
#                       weights (`αL, αR`).
#   * Matrix / Matrix — global kernel over `M` axis-aligned elements with
#                       homogeneous outer Dirichlet values `bL, bR`. Returns
#                       `(L_h u) / h²` with the chain-rule factor for the
#                       reference→physical map applied.
#
# Equation-free: this file computes the discrete Laplacian only. Initial
# conditions, time evolution, and any wave-equation-specific BCs are the
# downstream caller's responsibility.

################################################################################
# Per-element 1D Laplacian + SAT

# Pure-functional 1D Laplacian + SAT: loads `u` into an SVector and returns
# the result as an SVector, never touching the heap.
@inline function _apply_laplacian(u_s::SVector{N,T},
                                  gL, gR, gGL, gGR, αL, αR,
                                  ops::SBPOps{N,T}, τ) where {N, T}
    ΔuL  = u_s[1] - gL
    ΔuR  = u_s[N] - gR
    GuL  = dot(ops.G[1, :], u_s)
    GuR  = dot(ops.G[N, :], u_s)
    ΔGuL = GuL - gGL
    ΔGuR = GuR - gGR
    return ops.L * u_s + _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR, ops, τ)
end

# Convenience wrapper: caller supplies `u` and the neighbour data; we load
# to SVector, compute statically, write back into `Lu`.
function apply_laplacian!(Lu::AbstractVector, u::AbstractVector,
                          gL, gR, gGL, gGR, αL, αR;
                          ops::SBPOps{N}, τ) where {N}
    result = _apply_laplacian(SVector{N}(u), gL, gR, gGL, gGR, αL, αR, ops, τ)
    @inbounds for i in 1:N
        Lu[i] = result[i]
    end
    return Lu
end

################################################################################
# 1D global Laplacian
#
# `u`, `Lu` are (N, M) matrices: row = local GLL node, column = element.
# Boundary data `bL`, `bR` are scalars (outer Dirichlet values). The
# function name overloads the per-element kernel above on `AbstractMatrix`
# vs `AbstractVector`.
#
# Two implementations:
#
#   * CPU (`u isa Array`): SVector-based per-element loop. Fully
#     unrolled `SMatrix · SVector` algebra with no heap activity —
#     fastest CPU path.
#   * Non-CPU (Metal / CUDA / ROCm): KA kernel below
#     (`_laplacian1d_matrix_kernel!`), workgroup-per-element with N
#     workitems per workgroup. Each workitem computes one row of `L_h
#     · u` plus its share of the per-endpoint SAT lift.

using KernelAbstractions: @kernel, @index, @Const,
                          get_backend, synchronize

function apply_laplacian!(Lu::AbstractMatrix{T}, u::AbstractMatrix{T}, bL, bR;
                          dom, ops::SBPOps{N, T}, τ) where {N, T}
    M = size(Lu, 2)
    @assert size(u, 2) == M

    backend = get_backend(u)
    if backend isa KernelAbstractions.CPU
        return _apply_laplacian_1d_cpu!(Lu, u, T(bL), T(bR);
                                         dom, ops, τ = T(τ))
    end

    # Non-CPU backend: launch the KA kernel. Workgroup-per-element,
    # one workitem per local GLL node.
    inv_h2 = one(T) / T(dom.h)^2
    _laplacian1d_matrix_kernel!(backend, N)(
        Lu, u, ops, T(τ), T(bL), T(bR), inv_h2, Val(N);
        ndrange = N * M)
    return Lu
end

@inline function _apply_laplacian_1d_cpu!(Lu::AbstractMatrix{T},
                                           u::AbstractMatrix{T},
                                           bL::T, bR::T;
                                           dom, ops::SBPOps{N, T}, τ) where {N, T}
    M = size(Lu, 2)
    half = one(T) / 2

    @inbounds for m in 1:M
        u_self = SVector{N}(view(u, :, m))

        GuL_self = dot(ops.G[1, :], u_self)
        GuR_self = dot(ops.G[N, :], u_self)

        # Left face.
        if m == 1
            ΔuL  = u_self[1] - bL
            ΔGuL = zero(T)
            αL   = one(T)
        else
            u_left = SVector{N}(view(u, :, m-1))
            ΔuL  = u_self[1] - u_left[N]
            ΔGuL = GuL_self - dot(ops.G[N, :], u_left)
            αL   = half
        end

        # Right face — symmetric.
        if m == M
            ΔuR  = u_self[N] - bR
            ΔGuR = zero(T)
            αR   = one(T)
        else
            u_right = SVector{N}(view(u, :, m+1))
            ΔuR  = u_self[N] - u_right[1]
            ΔGuR = GuR_self - dot(ops.G[1, :], u_right)
            αR   = half
        end

        result = ops.L * u_self +
                 _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR, ops, τ)
        for i in 1:N
            Lu[i, m] = result[i]
        end
    end

    Lu .*= inv(dom.h^2)
    return Lu
end

# KA kernel — workgroup-per-element, N workitems each. Mirrors the
# CPU code's per-element flow but expressed in scalar form (each
# workitem computes one node of the per-element output). Loads `u_self`
# into shared memory so the volume L·u_self stencil and the boundary-
# trace gradients read from the workgroup-local copy instead of global
# memory.
#
# Uses the dense `Hinv[i, 1]` / `Hinv[i, N]` accessor in the SAT
# increment so the kernel works regardless of whether `Hinv` is stored
# as `Diagonal` (GLL branch) or `SMatrix` (Rational branch). For the
# Diagonal case this means 2·(N−1) zero mul-adds per workitem, which
# is negligible vs the memory-bound cost of the rest.
@kernel function _laplacian1d_matrix_kernel!(Lu::AbstractMatrix{T},
                                              @Const(u::AbstractMatrix{T}),
                                              ops, τ::T, bL::T, bR::T,
                                              inv_h2::T,
                                              ::Val{N}) where {T, N}
    m  = @index(Group, Linear)
    li = @index(Local, Linear)
    i  = li
    M  = size(u, 2)

    u_loc = @localmem T (N,)
    @inbounds u_loc[i] = u[i, m]
    @synchronize

    m  = @index(Group, Linear)
    li = @index(Local, Linear)
    i  = li

    half = T(1) / T(2)

    @inbounds u_self_1 = u_loc[1]
    @inbounds u_self_N = u_loc[N]

    # Endpoint gradients of `u_self`. Each workitem computes redundantly;
    # the result is broadcast across the workgroup, so we save one
    # @synchronize at the cost of two extra dot products per workitem.
    GuL_self = zero(T); GuR_self = zero(T)
    @inbounds for l in 1:N
        GuL_self += ops.G[1, l] * u_loc[l]
        GuR_self += ops.G[N, l] * u_loc[l]
    end

    # Left face SAT.
    if m == 1
        ΔuL  = u_self_1 - bL
        ΔGuL = zero(T)
        αL   = one(T)
    else
        @inbounds ΔuL = u_self_1 - u[N, m-1]
        GuL_left = zero(T)
        @inbounds for l in 1:N
            GuL_left += ops.G[N, l] * u[l, m-1]
        end
        ΔGuL = GuL_self - GuL_left
        αL = half
    end

    # Right face SAT.
    if m == M
        ΔuR  = u_self_N - bR
        ΔGuR = zero(T)
        αR   = one(T)
    else
        @inbounds ΔuR = u_self_N - u[1, m+1]
        GuR_right = zero(T)
        @inbounds for l in 1:N
            GuR_right += ops.G[1, l] * u[l, m+1]
        end
        ΔGuR = GuR_self - GuR_right
        αR = half
    end

    # Volume term: row `i` of `L · u_self`.
    s = zero(T)
    @inbounds for l in 1:N
        s += ops.L[i, l] * u_loc[l]
    end

    # Boundary-correction coefficients (shared across nodes in this
    # workgroup, computed redundantly to keep the per-workitem path
    # straight-line).
    cL = -αL * ΔuL
    cR =  αR * ΔuR
    b1 =  half * ΔGuL - τ * ΔuL
    bN = -half * ΔGuR - τ * ΔuR

    # SAT increment at node `i`. Dense Hinv form (correct for both
    # Diagonal and full-SMatrix Hinv).
    @inbounds inc = cL * ops.HinvG_L[i] + cR * ops.HinvG_R[i] +
                    b1 * ops.Hinv[i, 1] + bN * ops.Hinv[i, N]

    @inbounds Lu[i, m] = (s + inc) * inv_h2
end

################################################################################
# 1D first derivative with centred-flux SAT, driven by MeshConnectivity.
#
# ONE consistent `D` operator: reference SBP-G + centred-flux SAT at
# every interior element interface. With the SAT coefficient
# `1/(2 · Hphys_face)` the assembled operator `H · D` is exactly skew
# (`H·D + (H·D)ᵀ = 0`) on a periodic mesh — the SBP property
# `H·G = Q + ½(e_N e_Nᵀ − e_1 e_1ᵀ)` leaves `±½` diagonal boundary
# terms per element, and the centred-flux SAT cancels exactly those
# while coupling the neighbouring face values.
#
# Neighbour lookup goes through `geom.conn` (`neighbour`,
# `neighbour_face`) rather than hardwired `m ± 1`, so the same code
# serves uniform periodic lines and any future 1D mesh; this is also
# the structure the 2D/3D gradient/divergence SATs will reuse, with
# `orientation` transforms inserted at the lookup site.
#
# Faces with `bdry ≠ 0` (non-periodic outer boundaries) currently get
# *no* SAT contribution — a one-sided derivative. Dirichlet/Sommerfeld
# SAT variants hook in here in a later phase.

"""
    apply_D!(Du, u; geom::MeshGeometry{1, T, N}, ops::SBPOps{N, T}) → Du

Apply the consistent first-derivative operator (reference SBP-G +
centred-flux SAT at interior faces, neighbour relation from
`geom.conn`) to the state matrix `u :: (N, Ne)`, writing the physical
derivative into `Du`. Requires positively-oriented elements
(`handedness == +1`).
"""
function apply_D!(Du::AbstractMatrix{T}, u::AbstractMatrix{T};
                  geom::MeshGeometry{1, T, N}, ops::SBPOps{N, T}) where {N, T}
    Ne = geom.Ne
    @assert size(u) == size(Du) == (N, Ne)

    backend = get_backend(u)
    if backend isa KernelAbstractions.CPU
        return _apply_D_1d_cpu!(Du, u, geom, ops)
    end

    _apply_D_1d_kernel!(backend, N)(
        Du, u, ops, geom.conn.neighbour, geom.conn.bdry,
        geom.invjac, geom.Hphys, Val(N);
        ndrange = N * Ne)
    return Du
end

@inline function _apply_D_1d_cpu!(Du::AbstractMatrix{T}, u::AbstractMatrix{T},
                                  geom::MeshGeometry{1, T, N},
                                  ops::SBPOps{N, T}) where {N, T}
    Ne        = geom.Ne
    neighbour = geom.conn.neighbour
    bdry      = geom.conn.bdry
    half      = one(T) / 2

    @inbounds for m in 1:Ne
        u_self = SVector{N}(view(u, :, m))
        Gu = ops.G * u_self

        # Volume term: physical derivative via the per-node inverse
        # Jacobian (constant per element for affine line elements).
        for i in 1:N
            Du[i, m] = Gu[i] * geom.invjac[1, 1, i, m]
        end

        # Centred-flux SAT, coefficient 1/(2·Hphys_face). The 1D
        # neighbour-face convention: an interior face always meets the
        # *opposite* face of the neighbour, so the neighbour's trace
        # node is N at our face 1 and 1 at our face 2.
        if bdry[1, m] == 0
            mL = Int(neighbour[1, m])
            Du[1, m] += (u_self[1] - u[N, mL]) * half / geom.Hphys[1, m]
        end
        if bdry[2, m] == 0
            mR = Int(neighbour[2, m])
            Du[N, m] += (u[1, mR] - u_self[N]) * half / geom.Hphys[N, m]
        end
    end
    return Du
end

# KA kernel — workgroup-per-element, N workitems each; mirrors the CPU
# per-element flow in scalar form, with `u_self` staged through shared
# memory (same structure as `_laplacian1d_matrix_kernel!`).
@kernel function _apply_D_1d_kernel!(Du::AbstractMatrix{T},
                                     @Const(u::AbstractMatrix{T}),
                                     ops, @Const(neighbour), @Const(bdry),
                                     @Const(invjac), @Const(Hphys),
                                     ::Val{N}) where {T, N}
    m = @index(Group, Linear)
    i = @index(Local, Linear)

    u_loc = @localmem T (N,)
    @inbounds u_loc[i] = u[i, m]
    @synchronize

    m = @index(Group, Linear)
    i = @index(Local, Linear)

    half = T(1) / T(2)

    s = zero(T)
    @inbounds for l in 1:N
        s += ops.G[i, l] * u_loc[l]
    end
    @inbounds s *= invjac[1, 1, i, m]

    # Face SAT — only the two face workitems contribute.
    @inbounds if i == 1 && bdry[1, m] == 0
        mL = Int(neighbour[1, m])
        s += (u_loc[1] - u[N, mL]) * half / Hphys[1, m]
    end
    @inbounds if i == N && bdry[2, m] == 0
        mR = Int(neighbour[2, m])
        s += (u[1, mR] - u_loc[N]) * half / Hphys[N, m]
    end

    @inbounds Du[i, m] = s
end

################################################################################
# Diagnostic: assemble the global L_SAT matrix for `M` elements coupled
# DG-style. Used by the test suite to verify symmetry / null-space /
# spectrum properties; not on any simulation hot path.

function build_global_laplacian(M::Integer; ops, τ)
    G, L = ops.G, ops.L
    N = size(L, 1)

    T = eltype(L)
    n = N * M
    A = zeros(T, n, n)
    for j in 1:n
        e = zeros(T, n)
        e[j] = one(T)
        # precompute Gu per element
        Gu_all = [G * e[(i-1)*N+1 : i*N] for i in 1:M]
        for i in 1:M
            rng = (i-1)*N+1 : i*N
            gL  = i == 1 ? zero(T)             : e[first(rng) - 1]
            gR  = i == M ? zero(T)             : e[last(rng)  + 1]
            # outer mirror: ΔGu = 0 by using local value
            gGL = i == 1 ? Gu_all[i][begin]    : Gu_all[i-1][end]
            gGR = i == M ? Gu_all[i][end]      : Gu_all[i+1][begin]
            αL  = i == 1 ? one(T)              : one(T) / 2
            αR  = i == M ? one(T)              : one(T) / 2
            apply_laplacian!(view(A, rng, j), view(e, rng),
                             gL, gR, gGL, gGR, αL, αR; ops, τ=τ)
        end
    end
    return A
end
