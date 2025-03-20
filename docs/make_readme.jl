using Literate: Literate
using SparseArraysBase: SparseArraysBase

function ccq_logo(content)
  include_ccq_logo = """
    <img src="docs/src/assets/CCQ.png" width="20%" alt="Flatiron Center for Computational Quantum Physics logo.">
    """
  content = replace(content, "{CCQ_LOGO}" => include_ccq_logo)
  return content
end

Literate.markdown(
  joinpath(pkgdir(SparseArraysBase), "examples", "README.jl"),
  joinpath(pkgdir(SparseArraysBase));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
  preprocess=ccq_logo,
)
