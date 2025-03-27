using Literate: Literate
using SparseArraysBase: SparseArraysBase

function ccq_logo(content)
  include_ccq_logo = """
    <picture>
      <source media="(prefers-color-scheme: dark)" width="20%" srcset="docs/src/assets/CCQ-dark.png">
      <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="docs/src/assets/CCQ.png">
    </picture>
    """
  content = replace(content, "{CCQ_LOGO}" => include_ccq_logo)
  return content
end

Literate.markdown(
  joinpath(pkgdir(SparseArraysBase), "examples", "README.jl"),
  joinpath(pkgdir(SparseArraysBase));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
  postprocess=ccq_logo,
)
