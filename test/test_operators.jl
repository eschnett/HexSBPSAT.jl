# Tests for `src/operators.jl`: element / domain construction, `SBPOps`
# typing, and the SBP-property identities. (The polynomial-exactness
# assertions inside `_make_operators` already verify these in the Rational
# branch at construction time; here we add explicit external checks.)

using HexSBPSAT
using HexSBPSAT: make_element, make_domain, make_operators, SBPOps
using LinearAlgebra
using StaticArrays
using Test

# `_progress` is defined in `runtests.jl`. When this file is included
# stand-alone (e.g. via `include("test/test_operators.jl")`), define a
# local fallback that mirrors the runtests.jl one.
@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

@testset "operators" begin

    _progress("Rational element / SBP identities")
    @testset "make_element / make_domain (Rational)" begin
        # Rational branch: equispaced "vertex-centred" nodes on [0, 1].
        e = make_element(Rational{Int64}, 4)
        @test e.xs == 0//1 : 1//3 : 1//1
        @test e.h  == 1//3
    end

    @testset "SBPOps types and structure (Rational)" begin
        elem = make_element(Rational{Int64}, 4)
        ops  = make_operators(elem)
        @test ops isa SBPOps{4, Rational{Int64}, 16}
        # Rational H comes from the Vandermonde construction → dense SMatrix.
        @test ops.H isa SMatrix{4, 4, Rational{Int64}, 16}
        @test ops.G isa SMatrix{4, 4, Rational{Int64}, 16}
        @test ops.L isa SMatrix{4, 4, Rational{Int64}, 16}
    end

    @testset "SBP property H·D + (H·G)ᵀ = B (Rational, exact)" begin
        elem = make_element(Rational{Int64}, 4)
        ops  = make_operators(elem)
        lhs = Matrix(ops.H * ops.D) + Matrix(ops.H * ops.G)'
        @test lhs == Matrix(ops.B)
    end

    # Floating-point branches: Float64 + Float32 (GPU-friendly). Tolerances
    # scale with `eps(T)`; the constants below are sized to fit through N=17
    # (largest sweep entry) with comfortable margin.
    for T in (Float64, Float32)
        sbp_tol  = T === Float64 ? 1.0e-12 : 1.0e-4
        lap_tol  = T === Float64 ? 1.0e-12 : 1.0e-5

        _progress("$T GLL element, types, SBP sweep, L = D·G")
        @testset "make_element / make_domain ($T GLL)" begin
            # GLL collocation, h = minimum diff between nodes.
            ef = make_element(T, 4)
            @test eltype(ef.xs) === T
            @test length(ef.xs) == 4
            @test ef.xs[1]    ≈ zero(T)
            @test ef.xs[end]  ≈ one(T)
            @test ef.h        ≈ minimum(diff(ef.xs))
            # GLL nodes for N=4 cluster near the endpoints — first interior
            # gap is smaller than the central gap.
            @test diff(ef.xs)[1] < diff(ef.xs)[2]

            # Domain: N cell-centred elements covering [x0, x1].
            d = make_domain(T, 8, zero(T), one(T))
            @test d.N == 8
            @test d.h ≈ one(T)/8
            @test length(d.xs) == 8
        end

        @testset "SBPOps types and structure ($T GLL)" begin
            elem = make_element(T, 4)
            ops  = make_operators(elem)
            @test ops isa SBPOps{4, T, 16}
            # GLL H is diagonal — stored as a `Diagonal{T, SVector{4, T}}`.
            @test ops.H    isa Diagonal{T, SVector{4, T}}
            @test ops.Hinv isa Diagonal{T, SVector{4, T}}
        end

        @testset "SBP property H·D + (H·G)ᵀ ≈ B ($T GLL)" begin
            # Spans small/odd, GPU-friendly small power-of-2, and medium.
            # Larger N (e.g. 13, 17) compile expensively per specialisation;
            # the property is generic in N and Rational coverage above is
            # already exact, so further sizes give limited extra signal.
            for N in (3, 4, 8)
                elem = make_element(T, N)
                ops  = make_operators(elem)
                lhs = Matrix(ops.H * ops.D) + Matrix(ops.H * ops.G)'
                @test maximum(abs.(lhs - Matrix(ops.B))) < sbp_tol
            end
        end

        @testset "L = D·G ($T GLL)" begin
            # Identical to the Rational identity up to floating-point
            # roundoff — D·G has internal cancellations that perturb the
            # last few bits.
            elem = make_element(T, 4)
            ops  = make_operators(elem)
            @test maximum(abs.(ops.L - ops.D * ops.G)) < lap_tol
        end
    end

end
