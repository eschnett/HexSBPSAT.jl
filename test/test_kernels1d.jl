# Tests for the 1D SBP-SAT discrete Laplacian (`apply_laplacian!`,
# per-element + global), and the diagnostic `build_global_laplacian`.
# Wave-equation evolution tests live in WaveToySecondOrder's test
# suite — this file restricts itself to operator-level identities.

using HexSBPSAT
using HexSBPSAT: make_element, make_domain, make_operators,
                 apply_laplacian!, build_global_laplacian
using LinearAlgebra
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

@testset "kernels1d" begin

    _progress("operator-level identities (single + 2-element)")
    N, ns, x0, x1, h, xs = elem = make_element(Rational{Int64}, 5)
    ops = make_operators(elem)
    (; B, G, H, Hinv, HinvG_L, HinvG_R, D, L) = ops

    @testset "single-element SAT consistency" begin
        # With g matching boundary values and mirror gradients, SAT vanishes.
        for p in 0:N-1
            u = [x^p for x in xs]
            Gu = G * u
            Lu = similar(u)
            apply_laplacian!(Lu, u, u[begin], u[end], Gu[begin], Gu[end], 1, 1;
                             ops, τ=1)
            @test Lu == L * u
        end
    end

    @testset "single-element polynomial exactness" begin
        # For polynomials of degree ≤ N-1, L_SAT with exact Dirichlet data and
        # mirror gradients (ΔGu = 0) reproduces u''.
        for p in 0:N-1
            u = [x^p for x in xs]
            uxx = [p < 2 ? 0 : p*(p-1) * x^(p-2) for x in xs]
            gL  = x0^p
            gR  = x1^p
            Gu  = G * u
            Lu = similar(u)
            apply_laplacian!(Lu, u, gL, gR, Gu[begin], Gu[end], 1, 1;
                             ops, τ=1)
            @test Lu == uxx
        end
    end

    @testset "two-element coupling polynomial exactness" begin
        # Two elements stacked: [x0,x1] and [x1, 2x1-x0]. DG-style: distinct
        # nodes at the interface, coupled via SAT using neighbour edge values
        # and edge gradients.
        M = 2
        x_elem(i) = x0 .+ (i-1)*(x1-x0) .+ h*ns
        Xg = vcat((x_elem(i) for i in 1:M)...)
        for p in 0:N-1
            ug = [x^p for x in Xg]
            uxxg = [p < 2 ? 0 : p*(p-1) * x^(p-2) for x in Xg]
            # precompute Gu per element
            Gu_all = [G * ug[(i-1)*N+1 : i*N] for i in 1:M]
            result = similar(ug)
            for i in 1:M
                rng = (i-1)*N+1 : i*N
                ui  = ug[rng]
                gL  = i == 1 ? Xg[first(rng)]^p   : ug[first(rng) - 1]
                gR  = i == M ? Xg[last(rng)]^p    : ug[last(rng)  + 1]
                gGL = i == 1 ? Gu_all[i][begin]   : Gu_all[i-1][end]
                gGR = i == M ? Gu_all[i][end]     : Gu_all[i+1][begin]
                αL  = i == 1 ? 1//1 : 1//2
                αR  = i == M ? 1//1 : 1//2
                apply_laplacian!(view(result, rng), ui, gL, gR, gGL, gGR, αL, αR; ops, τ=1)
            end
            @test result == uxxg
        end
    end

    @testset "H·L_SAT is symmetric (interior SIPG + Nitsche outer)" begin
        # With SIPG at interior faces and full Nitsche at outer Dirichlet
        # faces, H·L_SAT is symmetric over the whole domain.
        for M in (2, 3, 4)
            for τ in (4//1, 16//1, 64//1)
                A = build_global_laplacian(M; ops, τ=τ)
                Hg = Matrix(kron(I(M), Matrix(H)))
                S = Hg * A
                @test maximum(abs.(S - S')) == 0
            end
        end
    end

    @testset "global L_SAT has no null space" begin
        # The SIPG coupling removes the kink-mode null space. Need τ above
        # threshold (here ≥ 64 for the M=2 case to make H·L_SAT spd).
        for M in (2, 3, 4)
            for τ in (64//1, 256//1)
                A = build_global_laplacian(M; ops, τ=τ)
                Af = Float64.(A)
                σmin = minimum(svdvals(Af))
                @test σmin > 1e-6
            end
        end
    end

    _progress("global SAT stability + jump penalty")
    @testset "global L_SAT wave-equation stability" begin
        # Eigenvalues real and non-positive for τ above threshold.
        for M in (2, 3, 4)
            for τ in (64//1, 256//1)
                A = build_global_laplacian(M; ops, τ=τ)
                λs = eigvals(Float64.(A))
                rel_tol = 1e-4 * maximum(abs.(λs))
                @test maximum(abs.(imag.(λs))) ≤ rel_tol
                @test maximum(real.(λs))       ≤ rel_tol
            end
        end
    end

    @testset "interface jump penalty contributes τ·jump² to energy" begin
        # Two elements; jump = 1 in u at the interface. The τ-jump contribution
        # to uᵀ H L_SAT u is −τ·(jump)².
        M = 2
        τ = 7//1
        A = build_global_laplacian(M; ops, τ=τ)
        Hg = Matrix(kron(I(M), Matrix(H)))
        u = zeros(Rational{Int}, N*M)
        u[N]   = 1//1
        u[N+1] = 0//1
        for Δτ in (1//1, 3//1, 5//1)
            A2 = build_global_laplacian(M; ops, τ=τ + Δτ)
            S1 = Hg * A
            S2 = Hg * A2
            e1 = u' * S1 * u
            e2 = u' * S2 * u
            @test e1 - e2 == Δτ * 1 * 1
        end
    end

end
