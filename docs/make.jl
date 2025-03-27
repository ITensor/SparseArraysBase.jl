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
    canonical="https://itensor.github.io/SparseArraysBase.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
  ),
  pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo="github.com/ITensor/SparseArraysBase.jl", devbranch="main", push_preview=true
)
