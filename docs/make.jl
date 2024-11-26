using SparseArraysBase: SparseArraysBase
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  SparseArraysBase, :DocTestSetup, :(using SparseArraysBase); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[SparseArraysBase],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="SparseArraysBase.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/SparseArraysBase.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/SparseArraysBase.jl", devbranch="main", push_preview=true
)
