# Tests for `apply_laplacian3d!` — the curvilinear-aware 3D SBP-DG
# Laplacian — and its diagnostics (`discrete_laplacian`,
# `spectral_radius_estimate`, `discrete_inner_product`,
# `discrete_l2_norm`). Equation-free: no wave-equation evolution, no
# Sommerfeld BC, no `Params3d`. End-to-end wave evolution tests live in
# WaveToySecondOrder's test suite.
#
# Each testset runs in both `Float64` and `Float32` to keep the GPU-
# friendly (Float32-only) path honest. The Metal smoke test at the end
# is gated on `HAS_METAL` so it skips silently on non-Apple-Silicon
# hosts.

using HexSBPSAT
using HexSBPSAT: make_element, make_operators, make_geometry,
                 apply_laplacian3d!, discrete_laplacian,
                 spectral_radius_estimate, discrete_inner_product,
                 discrete_l2_norm, physical_mass_diagonal, to_device
using HexMeshes: make_cubical_mesh, make_cubed_cube_mesh
using KernelAbstractions: CPU
using LinearAlgebra
using Random
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

# `Metal` and `CUDA` are weak deps of HexSBPSAT (`[weakdeps]` in
# Project.toml). Try to load each one; if it isn't installed (wrong
# platform / not added to the sandbox by `runtests.jl`) or not
# functional on this machine, the corresponding HAS_* gate stays
# false and the GPU testsets below skip silently.
const HAS_METAL = try
    @eval using Metal
    Metal.functional()
catch
    false
end

const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# When CUDA is functional, also probe the device's FP32:FP64 throughput
# ratio. The driver attribute `SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO` is
# 2 on workstation/datacenter GPUs (V100, A100, H100) with full half-
# speed FP64, 32+ on consumer Pascal/Turing/Ada/Ampere/Blackwell with
# crippled FP64. A ratio of `≤ 8` is the rough cutoff between "FP64 is
# reasonable" and "FP64 will take 30× longer than necessary". The
# Float64 CUDA testset below is gated on this so consumer cards still
# exercise the Float32 path without paying the slow-FP64 tax.
const HAS_CUDA_FP64 = HAS_CUDA && try
    ratio = CUDA.attribute(CUDA.device(),
                           CUDA.DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)
    ratio ≤ 8
catch
    # Attribute query failed (very old driver?) — be conservative and
    # skip the FP64 testset.
    false
end

@testset "apply_laplacian3d! (T=$T)" for T in (Float64, Float32)

    # Float32 has ~1e-7 round-off; relax assembled-matrix tolerances
    # accordingly. The qualitative correctness check (symmetry, NSD,
    # spectral-radius agreement) is identical for both precisions.
    sym_tol = T === Float64 ? 1.0e-9 : 1.0e-4
    nsd_tol = T === Float64 ? 1.0e-6 : 1.0e-2
    sr_rtol = T === Float64 ? 0.05  : 0.10
    ip_rtol = T === Float64 ? 1.0e-12 : 1.0e-5

    _progress("explicit Laplacian assembly: symmetric, NSD (T=$T)")
    @testset "discrete_laplacian: symmetric and negative-semi-definite (T=$T)" begin
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2

        L = Matrix(discrete_laplacian(geom, ops, τ))
        Hd = physical_mass_diagonal(geom, ops)
        S  = Diagonal(Hd) * L

        @test maximum(abs.(S - S')) < sym_tol      # mass-symmetric
        # Symmetrise to avoid hint-of-asymmetry warnings; then NSD up to round-off.
        λs = eigvals(Symmetric((S + S') / 2))
        @test maximum(λs) < nsd_tol
    end

    _progress("spectral_radius_estimate: matches max|λ| of assembled L (T=$T)")
    @testset "spectral_radius_estimate against full spectrum (cubical, T=$T)" begin
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2

        L  = Matrix(discrete_laplacian(geom, ops, τ))
        λ_dense = maximum(abs, eigvals(L))
        Random.seed!(20260528)
        λ_iter  = spectral_radius_estimate(geom, ops, τ)
        @test isapprox(λ_iter, λ_dense; rtol = sr_rtol)
    end

    _progress("apply_laplacian3d! tag-7 free face contributes nothing (T=$T)")
    @testset "tag-7 face is a free face (T=$T)" begin
        # Inject a wave-equation-style tag-7 on every outer face and verify
        # that apply_laplacian3d!'s output:
        #   1. differs from the Dirichlet-default version (tag-7 = no SAT)
        #   2. stays finite (clean no-op, not NaN-producing skip).
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2

        u   = randn(T, N, N, N, mesh.Ne)
        ü_d = similar(u)
        apply_laplacian3d!(ü_d, u, ntuple(_ -> zero(T), Val(6)); geom, ops, τ)

        # Re-tag every outer face as `7` and re-evaluate.
        for e in 1:mesh.Ne, f in 1:6
            if geom.conn.neighbour[f, e] == 0
                geom.conn.bdry[f, e] = Int8(7)
            end
        end
        ü_s = similar(u)
        apply_laplacian3d!(ü_s, u, ntuple(_ -> zero(T), Val(6)); geom, ops, τ)

        @test ü_d != ü_s
        @test all(isfinite, ü_s)
    end

    _progress("free face implements the natural boundary lift (T=$T)")
    @testset "free face: natural boundary lift recovers L_h(x²) = 2 (T=$T)" begin
        # Structural test for the "free face" face SAT: on tag-7 (free)
        # outer faces, the face SAT must contribute the *natural* boundary
        # lift `wF · Gn_self` (the discrete analog of the IBP term
        # `∮ψ · ∂_n u dS`), not zero. Without this contribution the
        # operator is implicitly Neumann-zero at outer faces, and the
        # Sommerfeld SAT layered on top is non-dissipative.
        #
        # Verification: on an axis-aligned cubical mesh the SBP operator
        # is exact for polynomials of degree ≤ N-1. For u = x²
        # (degree 2, with N ≥ 3), L_h(u) must equal ∇²(x²) = 2 at every
        # node — *including* outer-face nodes whose contribution depends
        # on the natural lift being present.
        N    = 3
        M    = 4
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, -one(T), one(T))   # [-1, 1]³
        # Re-tag every outer face as Sommerfeld (7) to exercise the
        # free-face SAT path.
        for e in 1:mesh.Ne, f in 1:6
            if mesh.conn.neighbour[f, e] == 0
                mesh.conn.bdry[f, e] = Int8(7)
            end
        end
        geom = make_geometry(mesh, elem)
        u = T.(geom.coords[1, :, :, :, :] .^ 2)
        ü = similar(u)
        τ = T(3//2) * (N - 1)^2
        apply_laplacian3d!(ü, u, ntuple(_ -> zero(T), Val(6)); geom, ops, τ)
        # Exact in floating-point arithmetic for this polynomial degree.
        @test maximum(abs, ü .- T(2)) < 100 * eps(T)
    end

    _progress("discrete_inner_product + discrete_l2_norm (T=$T)")
    @testset "inner product / L² norm consistency (T=$T)" begin
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)

        u = ones(T, N, N, N, mesh.Ne)
        v = ones(T, N, N, N, mesh.Ne)
        ip = discrete_inner_product(u, v, geom, ops)
        @test isapprox(ip, sum(geom.Hphys); rtol = ip_rtol)
        @test isapprox(ip, one(T); rtol = ip_rtol)   # volume of the unit cube

        nrm = discrete_l2_norm(u, geom, ops)
        @test isapprox(nrm, sqrt(ip); rtol = ip_rtol)
    end

    _progress("to_device CPU round-trip (T=$T)")
    @testset "to_device round-trip on CPU backend (T=$T)" begin
        # Round-trip preserves bit-identical kernel output on CPU. Cheap
        # smoke test of the migration path.
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2

        geom_dev = to_device(geom, CPU())
        @test geom_dev.coords ≈ geom.coords
        @test geom_dev !== geom

        u  = randn(T, N, N, N, mesh.Ne)
        bdry = ntuple(_ -> zero(T), Val(6))
        ü_host = similar(u); ü_dev = similar(u)
        apply_laplacian3d!(ü_host, u, bdry; geom,            ops, τ)
        apply_laplacian3d!(ü_dev,  u, bdry; geom = geom_dev, ops, τ)
        @test ü_host == ü_dev
    end

end

# ───────────── Curvilinear SIPG penalty threshold ─────────────
#
# Regression test for the docstring claim that the SIPG penalty
# `τ ≈ 8·(N-1)²` is sufficient on curvilinear / multi-patch meshes
# (cubed cube, inflated cube). At lower τ the bare `apply_laplacian3d!`
# operator has growing modes at higher (N, M) — even with Dirichlet
# outer BC and no Sommerfeld pass — that manifest as exponential blowup
# under symplectic time-stepping.
#
# This was the second bug uncovered after the Sommerfeld investigation:
# at N=4 M=6 inflated cube with τ = 1.5·(N-1)² = 13.5 (the cubical-mesh
# value), random-IC leapfrog blows up by 6+ orders of magnitude in
# ~1000 steps. With τ = 8·(N-1)² = 72 it stays bounded.
@testset "Inflated cube curvilinear penalty: τ = 8·(N-1)² stabilises N=4 M=6" begin
    T = Float64
    N = 4
    M = 6
    elem = make_element(T, N)
    ops  = make_operators(elem)
    # Dirichlet outer (no Sommerfeld pass) — Bug 2 is in the bare
    # `apply_laplacian3d!`, not in the Sommerfeld SAT layered on top.
    mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), M)
    geom = make_geometry(mesh, elem)
    τ = T(8) * (N - 1)^2

    bdry = ntuple(_ -> zero(T), Val(6))
    # `spectral_radius_estimate` consumes the RNG state, so seed for it
    # first; then reseed for the IC so the result is reproducible.
    Random.seed!(20260601)
    λ_max = spectral_radius_estimate(geom, ops, τ)
    dt = T(1//2) * T(2) / sqrt(λ_max)   # Störmer–Verlet limit · 0.5 safety

    Random.seed!(20260601)
    u  = randn(T, N, N, N, mesh.Ne)
    u̇  = randn(T, N, N, N, mesh.Ne)
    ü  = similar(u)

    # Discrete Hamiltonian E = ½⟨u̇, u̇⟩_H + ½⟨u, −L_h u⟩_H. For a NSD
    # `L_h` and symplectic integration this is conserved within an
    # O(dt^p) modified-Hamiltonian envelope. Bug 2 (τ too low) makes
    # `L_h` indefinite — `V` goes negative for the unstable eigenmode
    # and the energy grows without bound.
    function energy(u, u̇)
        Lu = similar(u)
        apply_laplacian3d!(Lu, u, bdry; geom, ops, τ)
        K = discrete_inner_product(u̇, u̇, geom, ops) / 2
        V = -discrete_inner_product(u, Lu, geom, ops) / 2
        return K, V, K + V
    end
    K0, V0, E0 = energy(u, u̇)
    @test V0 > 0   # `L_h` is NSD ⇒ ⟨u, −L_h u⟩ ≥ 0 for any u

    # Manual leapfrog (Störmer–Verlet) using only `apply_laplacian3d!`
    # — no OrdinaryDiffEq / wave-specific `recommended_dt` dependency.
    apply_laplacian3d!(ü, u, bdry; geom, ops, τ)
    u̇ .+= (dt / 2) .* ü
    n_steps = 100
    for _ in 1:n_steps
        u .+= dt .* u̇
        apply_laplacian3d!(ü, u, bdry; geom, ops, τ)
        u̇ .+= dt .* ü
    end
    u̇ .-= (dt / 2) .* ü   # align u̇ back to integer step for energy diag

    @test all(isfinite, u) && all(isfinite, u̇)
    _, _, E_end = energy(u, u̇)
    # Energy conservation to ~15% (generous margin for modified-Ham
    # drift at cfl=0.5). The bug produces 6+ orders of magnitude growth.
    @test abs(E_end - E0) < T(0.15) * E0
end

# ───────────────────────────── Metal ─────────────────────────────
#
# GPU smoke test on Metal. Gated on `Metal.functional()` so the test
# silently skips on machines without an Apple GPU (Linux/x86 CI runners
# included). Verifies that the full operator chain works end-to-end on
# the GPU:
#   • `to_device(geom, MetalBackend())` migrates geometry
#   • state allocated as `MtlArray{Float32}`
#   • `apply_laplacian3d!` runs on Metal
#   • `discrete_inner_product` runs on Metal (GPUArrays mapreduce)
#   • `spectral_radius_estimate` runs on Metal (KrylovKit + matrix-free)
# and that the final result matches the host path within Float32
# round-off.
if HAS_METAL
    @testset "apply_laplacian3d! on Metal (Float32)" begin
        _progress("Metal: to_device + apply_laplacian3d! match CPU")
        T    = Float32
        N    = 4
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2
        bdry = ntuple(_ -> zero(T), Val(6))

        # Host reference.
        Random.seed!(20260528)
        u_host  = randn(T, N, N, N, mesh.Ne)
        ü_host  = similar(u_host)
        apply_laplacian3d!(ü_host, u_host, bdry; geom, ops, τ)

        # Migrate geometry + state to Metal.
        backend  = MetalBackend()
        geom_dev = to_device(geom, backend)
        u_dev    = MtlArray(u_host)
        ü_dev    = MtlArray(zeros(T, N, N, N, mesh.Ne))

        apply_laplacian3d!(ü_dev, u_dev, bdry; geom = geom_dev, ops, τ)
        @test Array(ü_dev) ≈ ü_host

        _progress("Metal: discrete_inner_product matches CPU")
        # ⟨u, u⟩_{H_phys} on Metal — should match host within Float32 round-off.
        ip_host = discrete_inner_product(u_host, u_host, geom, ops)
        ip_dev  = discrete_inner_product(u_dev,  u_dev,  geom_dev, ops)
        @test isapprox(ip_dev, ip_host; rtol = sqrt(eps(T)))

        _progress("Metal: spectral_radius_estimate runs to a finite positive")
        # spectral_radius_estimate uses Random.randn! into a device array
        # and KrylovKit on a matrix-free shell — exercises the full GPU
        # stack. Different host vs device RNG, so don't expect
        # bit-identity; just same order of magnitude.
        Random.seed!(20260528)
        λ_host = spectral_radius_estimate(geom,     ops, τ)
        Random.seed!(20260528)
        λ_dev  = spectral_radius_estimate(geom_dev, ops, τ)
        @test isfinite(λ_dev) && λ_dev > 0
        @test 0.5f0 * λ_host < λ_dev < 2 * λ_host
    end
end

# ───────────────────────────── CUDA ─────────────────────────────
#
# GPU smoke test on CUDA, run for both Float32 and Float64. Gated on
# `CUDA.functional()` so the testset silently skips on machines without
# an NVIDIA GPU (including macOS dev hosts and most CI runners). The
# Float64 portion is *additionally* gated on the device's FP32:FP64
# perf-ratio attribute — on consumer cards with crippled FP64 we run
# only the Float32 testset and skip Float64 to keep CI tractable.
#
# Verifies the same operator chain the Metal testset checks:
#   • `to_device(geom, CUDABackend())` migrates geometry
#   • state allocated as `CuArray{T}`
#   • `apply_laplacian3d!` runs on CUDA
#   • `discrete_inner_product` runs on CUDA (GPUArrays mapreduce)
#   • `spectral_radius_estimate` runs on CUDA (KrylovKit + matrix-free)
#
# This file has never been exercised on a real CUDA machine — the
# package is developed on Apple Silicon. If a test fails, the most
# likely causes are (a) a CUDA-specific issue with our `to_device`
# implementation (which uses `KernelAbstractions.allocate(backend, …)`
# and `copyto!`, both backend-agnostic in principle) or (b) a tolerance
# mismatch on cards with non-IEEE FMA semantics.
if HAS_CUDA
    cuda_types = HAS_CUDA_FP64 ? (Float32, Float64) : (Float32,)
    @testset "apply_laplacian3d! on CUDA (T=$T)" for T in cuda_types
        _progress("CUDA: to_device + apply_laplacian3d! match CPU (T=$T)")
        # Sized for cheap CI: a single 2×2×2 cube of N=4 GLL nodes per
        # element. The host reference is computed first so any device
        # divergence shows up directly.
        N    = 4
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        τ    = T(3//2) * (N - 1)^2
        bdry = ntuple(_ -> zero(T), Val(6))

        # Host reference.
        Random.seed!(20260528)
        u_host  = randn(T, N, N, N, mesh.Ne)
        ü_host  = similar(u_host)
        apply_laplacian3d!(ü_host, u_host, bdry; geom, ops, τ)

        # Migrate geometry + state to CUDA.
        backend  = CUDABackend()
        geom_dev = to_device(geom, backend)
        u_dev    = CuArray(u_host)
        ü_dev    = CuArray(zeros(T, N, N, N, mesh.Ne))

        apply_laplacian3d!(ü_dev, u_dev, bdry; geom = geom_dev, ops, τ)
        # GPUs are free to issue FMAs that the CPU path may have
        # separated; allow a tight ULP-scaled tolerance instead of
        # bit-equality. Float64 → 1e-12, Float32 → ~1e-6.
        @test isapprox(Array(ü_dev), ü_host;
                       atol = 100 * eps(T) * maximum(abs, ü_host),
                       rtol = 100 * eps(T))

        _progress("CUDA: discrete_inner_product matches CPU (T=$T)")
        ip_host = discrete_inner_product(u_host, u_host, geom, ops)
        ip_dev  = discrete_inner_product(u_dev,  u_dev,  geom_dev, ops)
        @test isapprox(ip_dev, ip_host; rtol = sqrt(eps(T)))

        _progress("CUDA: spectral_radius_estimate runs to a finite positive (T=$T)")
        # Different host vs device RNG → don't expect bit-identity,
        # just same order of magnitude.
        Random.seed!(20260528)
        λ_host = spectral_radius_estimate(geom,     ops, τ)
        Random.seed!(20260528)
        λ_dev  = spectral_radius_estimate(geom_dev, ops, τ)
        @test isfinite(λ_dev) && λ_dev > 0
        @test T(0.5) * λ_host < λ_dev < 2 * λ_host
    end
end
