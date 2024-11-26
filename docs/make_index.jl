using Literate: Literate
using SparseArraysBase: SparseArraysBase

Literate.markdown(
  joinpath(pkgdir(SparseArraysBase), "examples", "README.jl"),
  joinpath(pkgdir(SparseArraysBase), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
