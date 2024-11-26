using Literate: Literate
using SparseArraysBase: SparseArraysBase

Literate.markdown(
  joinpath(pkgdir(SparseArraysBase), "examples", "README.jl"),
  joinpath(pkgdir(SparseArraysBase));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
