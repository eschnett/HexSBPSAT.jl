# Precision-genericity tests for HexSBPSAT.
#
# Goal: catch accidental `Float64` use (or any other type-promotion
# pollution) when the package is exercised at non-default precisions.
# The tests check `eltype` invariants — if any intermediate computation
# accidentally goes through `Float64`, the final array's `eltype` will
# be `Float64` instead of the requested `T`, and the `@test eltype(…) === T`
# line will trip.
#
# Precisions covered:
#
# * `Float32`     — native IEEE single precision.
# * `Float64`     — native IEEE double precision.
# * `Float32x2`   — double-single (MultiFloats), eps ≈ `1.4e-14`.
# * `Float64x2`   — double-double (MultiFloats), eps ≈ `4.9e-32`.
#
# All four are `isbits` floating-point types, so all paths including
# the KernelAbstractions kernel (`apply_laplacian!`) are exercised.
# `BigFloat` is *not* tested: it is heap-allocated and incompatible
# with `@localmem`.
#
# `MultiFloats.Float64x2` / `Float32x2` go through a Float64 fallback
# in `_gll_basis` (see `operators.jl`) because PolynomialBases'
# `LobattoLegendre` Newton iteration uses `cos`, which MultiFloats
# does not define. The fallback converts a Float64-built basis to T;
# the SBP operator entries inherit Float64 accuracy (~1.5e-8), but
# kernel arithmetic runs at full T precision.

using HexSBPSAT
using HexSBPSAT: eigsolve_default_tol
using HexMeshes: make_uniform_hex
using MultiFloats: Float32x2, Float64x2
using Test

const _PRECISIONS = (Float32, Float64, Float32x2, Float64x2)

@testset "precision-genericity (T ∈ Float32, Float64, Float32x2, Float64x2)" begin

    @testset "eigsolve_default_tol(T) = eps(T)^(1/4)" begin
        for T in _PRECISIONS
            @test eigsolve_default_tol(T) == eps(T)^(1//4)
        end
        # Sanity ordering: Float32 loose, Float64 tighter, MultiFloats much tighter.
        @test eigsolve_default_tol(Float64)   < eigsolve_default_tol(Float32)
        @test eigsolve_default_tol(Float32x2) < eigsolve_default_tol(Float32)
        @test eigsolve_default_tol(Float64x2) < eigsolve_default_tol(Float64)
    end

    @testset "make_element / make_operators: types are pure" begin
        for T in _PRECISIONS
            elem = make_element(T, 4)
            @test eltype(elem.xs) === T
            @test typeof(elem.h)  === T
            @test typeof(elem.x0) === T
            @test typeof(elem.x1) === T

            ops = make_operators(elem)
            @test ops isa SBPOps{4, T}
            @test eltype(ops.B)       === T
            @test eltype(ops.G)       === T
            @test eltype(ops.H)       === T
            @test eltype(ops.Hinv)    === T
            @test eltype(ops.HinvG_L) === T
            @test eltype(ops.HinvG_R) === T
            @test eltype(ops.D)       === T
            @test eltype(ops.L)       === T
        end
    end

    @testset "make_geometry: types are pure" begin
        for T in _PRECISIONS
            mesh = make_uniform_hex(T, 2, T(0), T(1))
            elem = make_element(T, 3)
            geom = make_geometry(mesh, elem)

            @test geom isa MeshGeometry{3, T, 3}
            @test eltype(geom.coords)     === T
            @test eltype(geom.jac)        === T
            @test eltype(geom.invjac)     === T
            @test eltype(geom.detjac)     === T
            @test eltype(geom.Hphys)      === T
            @test eltype(geom.handedness) === Int8     # by design

            # Workspace is separate from MeshGeometry (step 6); check the
            # face_trace buffer's eltype the same way.
            work = HexSBPSAT.make_workspace(geom)
            @test work isa HexSBPSAT.MeshWorkspace{3, T, 3}
            @test eltype(work.face_trace) === T
        end
    end

    @testset "diagnostics: discrete_inner_product / l2_norm / mass_diagonal" begin
        for T in _PRECISIONS
            mesh = make_uniform_hex(T, 2, T(0), T(1))
            elem = make_element(T, 3)
            geom = make_geometry(mesh, elem)
            ops  = make_operators(elem)

            u = fill(T(2), 3, 3, 3, mesh.Ne)
            v = fill(T(3), 3, 3, 3, mesh.Ne)
            atol = T(1e-5)

            ip = discrete_inner_product(u, v, geom, ops)
            @test typeof(ip) === T
            # ∫ 2 · 3 dx over [0,1]³ = 6
            @test ip ≈ T(6) atol = atol

            nrm = discrete_l2_norm(u, geom, ops)
            @test typeof(nrm) === T
            # ‖2‖₂² = ∫ 4 dx = 4 ⇒ ‖2‖₂ = 2
            @test nrm ≈ T(2) atol = atol

            md = physical_mass_diagonal(geom, ops)
            @test eltype(md) === T
            @test length(md) == 3 * 3 * 3 * mesh.Ne
            # Σ Hphys = ∫ 1 dx = 1
            @test sum(md) ≈ T(1) atol = atol
        end
    end

    @testset "apply_laplacian! (all precisions)" begin
        # All four types are isbits, so they pass through `@localmem`
        # cleanly. The MultiFloats path additionally exercises the
        # Float64 → T conversion in `_gll_basis`.
        for T in _PRECISIONS
            mesh = make_uniform_hex(T, 2, T(0), T(1))
            elem = make_element(T, 3)
            geom = make_geometry(mesh, elem)
            ops  = make_operators(elem)
            work = HexSBPSAT.make_workspace(geom)

            u  = randn(T, 3, 3, 3, mesh.Ne)
            ü  = similar(u)
            bdry = ntuple(_ -> zero(T), Val(6))
            apply_laplacian!(ü, u, bdry; geom, ops, work, τ = T(2))
            @test eltype(ü) === T
            @test all(isfinite, ü)
        end
    end

    @testset "spectral_radius_estimate (Float32 / Float64 only)" begin
        # KrylovKit's tridiagonal eigensolver dispatches to LAPACK
        # and only supports Union{Float32, Float64, ComplexF32, ComplexF64}.
        # MultiFloats precisions cannot use this diagnostic; document.
        for T in (Float32, Float64)
            mesh = make_uniform_hex(T, 2, T(0), T(1))
            elem = make_element(T, 3)
            geom = make_geometry(mesh, elem)
            ops  = make_operators(elem)
            ω² = spectral_radius_estimate(geom, ops, T(2))
            @test typeof(ω²) === T
            @test isfinite(ω²)
            @test ω² > 0
        end
    end

    @testset "spectral_radius_estimate unsupported on MultiFloats (documented)" begin
        # Confirm the limitation surfaces as a MethodError so callers
        # can pattern-match if needed; remove this test if KrylovKit
        # ever gains MultiFloats support.
        for T in (Float32x2, Float64x2)
            mesh = make_uniform_hex(T, 2, T(0), T(1))
            elem = make_element(T, 3)
            geom = make_geometry(mesh, elem)
            ops  = make_operators(elem)
            @test_throws MethodError spectral_radius_estimate(geom, ops, T(2))
        end
    end

    @testset "MultiFloats: cos(::MultiFloat) is still broken (canary)" begin
        # `HexSBPSATMultiFloatsExt` opts MultiFloats out of the native
        # `LobattoLegendre` build because `cos(::MultiFloat)` is defined
        # but throws an "instructive error" inviting the user to
        # `MultiFloats.use_bigfloat_transcendentals()`. If MultiFloats
        # ever ships a working `cos`, these `@test_throws` lines stop
        # tripping — the test fails, and that's the cue to delete
        # `ext/HexSBPSATMultiFloatsExt.jl` and its [weakdeps] /
        # [extensions] entries in `Project.toml`.
        @test_throws Exception cos(one(Float32x2))
        @test_throws Exception cos(one(Float64x2))
        # And confirm the extension's override is in effect.
        @test HexSBPSAT._supports_lobatto_native(Float32x2) == false
        @test HexSBPSAT._supports_lobatto_native(Float64x2) == false
        # The default (and the Float32 / Float64 cases) should be true.
        @test HexSBPSAT._supports_lobatto_native(Float32) == true
        @test HexSBPSAT._supports_lobatto_native(Float64) == true
    end

end
