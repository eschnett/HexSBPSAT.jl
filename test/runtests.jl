# Per-platform GPU backend installs into the test sandbox. The HAS_*
# gates inside `test_apply_laplacian3d.jl` cause the corresponding
# testsets to skip silently whenever the backend isn't installed or
# `functional()` reports false (e.g. CI runners without GPUs), so the
# install only needs to happen on platforms where the backend has any
# chance of working.
if Sys.isapple() && Sys.ARCH === :aarch64
    # Apple Silicon → Metal. Metal.jl installs cleanly only on
    # apple/aarch64; on every other platform we leave the sandbox alone.
    using Pkg
    Pkg.add("Metal")
elseif !Sys.isapple()
    # Linux / Windows → CUDA. CUDA.jl installs on x86_64 + aarch64 Linux
    # and on Windows; skip macOS entirely (no consumer Mac ships with an
    # NVIDIA GPU on Apple Silicon, and the artifacts for x86 macOS are
    # not maintained). The `HAS_CUDA` gate inside the test file handles
    # "installed but no functional GPU" on Linux CI runners.
    using Pkg
    Pkg.add("CUDA")
end

using Test
using HexSBPSAT

function _section(label)
    printstyled(stderr, "── ", label, " ──\n"; color = :cyan, bold = true)
    flush(stderr)
end

_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                  flush(stderr))

@testset verbose = true "HexSBPSAT" begin
    _section("test_operators.jl");          include("test_operators.jl")
    _section("test_kernels1d.jl");          include("test_kernels1d.jl")
    _section("test_apply_D1d.jl");          include("test_apply_D1d.jl")
    _section("test_geometry.jl");           include("test_geometry.jl")
    _section("test_apply_laplacian3d.jl");  include("test_apply_laplacian3d.jl")
    _section("test_apply_laplacian2d.jl");  include("test_apply_laplacian2d.jl")
    _section("test_precision.jl");          include("test_precision.jl")
    _section("test_periodic.jl");           include("test_periodic.jl")
end
