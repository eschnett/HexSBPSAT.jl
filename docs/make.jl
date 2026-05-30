# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using HexSBPSAT

makedocs(; sitename="HexSBPSAT", format=Documenter.HTML(), modules=[HexSBPSAT])

deploydocs(; repo="github.com/eschnett/HexSBPSAT.jl.git", devbranch="main", push_preview=true)
