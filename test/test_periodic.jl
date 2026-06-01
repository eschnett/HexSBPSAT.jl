using HexMeshes: make_uniform_hex, make_uniform_quad, make_uniform_line
using HexSBPSAT
using Test

# Smoke tests that `apply_laplacian!` (SIPG) does the right thing when
# fed a periodic mesh from HexMeshes. The defining property of a
# correctly-glued periodic Laplacian is that the constant function
# lives in its kernel — there's no Dirichlet face data fighting it at
# the outer boundary because every face is now an interior seam.

@testset "periodic mesh: apply_laplacian! (3D SIPG)" begin
    _progress("3D periodic constant-in-kernel")
    T = Float64; N = 3; τ = 1.5 * (N - 1)^2
    elem = make_element(T, N)
    ops  = make_operators(elem)

    # Periodic cube: L·1 == 0 exactly (every face is an interior seam,
    # centred flux drops out).
    mesh_per = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
    geom_per = make_geometry(mesh_per, elem)
    work_per = make_workspace(geom_per)
    u = fill(one(T), N, N, N, geom_per.Ne)
    ü = similar(u)
    apply_laplacian!(ü, u, ntuple(_ -> zero(T), Val(6));
                       geom = geom_per, ops = ops, work = work_per, τ = τ)
    @test maximum(abs, ü) < 1e-12

    # Sanity: non-periodic mesh with Dirichlet u_face=0 must *not* have
    # the constant in its kernel — the SIPG penalty against u_face = 0
    # forces a large interior contribution at the boundary layer.
    mesh_dir = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0)
    geom_dir = make_geometry(mesh_dir, elem)
    work_dir = make_workspace(geom_dir)
    u  = fill(one(T), N, N, N, geom_dir.Ne)
    ü = similar(u)
    apply_laplacian!(ü, u, ntuple(_ -> zero(T), Val(6));
                       geom = geom_dir, ops = ops, work = work_dir, τ = τ)
    @test maximum(abs, ü) > 1.0
end

@testset "periodic mesh: apply_laplacian! (2D SIPG)" begin
    _progress("2D periodic constant-in-kernel")
    T = Float64; N = 3; τ = 1.5 * (N - 1)^2
    elem = make_element(T, N)
    ops  = make_operators(elem)

    mesh_per = make_uniform_quad(T, 3, 2, 0.0, 1.0; periodic = true)
    geom_per = make_geometry(mesh_per, elem)
    work_per = make_workspace(geom_per)
    u  = fill(one(T), N, N, geom_per.Ne)
    ü = similar(u)
    apply_laplacian!(ü, u, ntuple(_ -> zero(T), Val(4));
                       geom = geom_per, ops = ops, work = work_per, τ = τ)
    @test maximum(abs, ü) < 1e-12
end
