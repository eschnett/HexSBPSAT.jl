# Tests for `HexSBPSAT/src/geometry.jl`: the operator-aware
# `MeshGeometry`, `make_geometry`, and `to_device` round-trips on the
# CPU backend. The analytic-Jacobian / outer-sphere-exactness
# assertions test that `make_geometry` composes the `HexMeshes`
# parametric patch maps correctly with the GLL collocation grid when
# the mesh's `patch_desc` carries `Shell` and `Inflation` entries.

using HexSBPSAT
using HexSBPSAT: make_element, make_operators, make_geometry
using HexMeshes: make_inflated_cube_mesh, Shell
using Test

# `_progress` is defined in `runtests.jl`; fallback for stand-alone runs.
@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

@testset "geometry" begin

    _progress("inflated cube + make_geometry: analytic Jacobian is well-formed")
    @testset "make_inflated_cube_mesh: geometry is well-formed (M=3, N=4)" begin
        T = Float64
        R2_expected = 4.0
        m    = make_inflated_cube_mesh(T, 1.0, 2.0, R2_expected, 3)
        elem = make_element(T, 4)
        g    = make_geometry(m, elem)

        @test size(g.coords) == (3, 4, 4, 4, m.Ne)
        @test !any(isnan, g.coords)
        @test !any(isnan, g.jac)
        @test all(>(0), g.detjac)
        # Right-handed local frames everywhere by construction.
        @test all(==(1), g.handedness)

        # Shell-patch nodes lie inside the ball |x| ≤ R2 (up to FP slack).
        r_all = sqrt.(g.coords[1, :, :, :, :].^2 .+
                      g.coords[2, :, :, :, :].^2 .+
                      g.coords[3, :, :, :, :].^2)
        @test maximum(r_all) ≤ R2_expected + 1e-8

        # Shell-element outermost-radial face nodes lie on the outer
        # sphere exactly. A shell element on the outer face is one whose
        # patch kind is `Shell` and whose `patch_idx[1]` equals the
        # patch's radial element count `dims[1]` (i.e. the last radial
        # element of that patch).
        shell_outer_max_err = 0.0
        for e in 1:m.Ne
            pd = m.patch_desc[m.patch_id[e]]
            pd.kind === Shell || continue
            idx_radial = Int(m.patch_idx[1, e])
            idx_radial == pd.shell.dims[1] || continue
            # Reference-cube ξ = 1 (radial axis) corresponds to a = a_hi
            # = 1 inside this last radial element ⇒ on R2.
            for k in 1:4, j in 1:4
                r = sqrt(g.coords[1, 4, j, k, e]^2 +
                         g.coords[2, 4, j, k, e]^2 +
                         g.coords[3, 4, j, k, e]^2)
                shell_outer_max_err = max(shell_outer_max_err, abs(r - R2_expected))
            end
        end
        @test shell_outer_max_err < 1e-12
    end

end
