[![Build Status](https://travis-ci.org/JuliaInv/Multigrid.jl.svg?branch=master)](https://travis-ci.org/JuliaInv/Multigrid.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaInv/Multigrid.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaInv/Multigrid.jl?branch=master)
[![Build status](https://ci.appveyor.com/api/projects/status/itta987m129uroku?svg=true)](https://ci.appveyor.com/project/lruthotto/multigrid-jl)

# Multigrid.jl

A multigrid package in Julia. Uses shared memory parallelism using OMP and [`ParSpMatVec`] (https://github.com/JuliaInv/ParSpMatVec.jl) .

Includes:

1) Geometric multigrid on a regular mesh.

2) Smoothed Aggregation AMG Multigrid, based on the following paper (please cite if you use this code):

   Eran Treister and Irad Yavneh, Non-Galerkin Multigrid based on Sparsified Smoothed Aggregation. SIAM Journal on Scientific Computing, 37 (1), A30-A54, 2015.

Options for V,F,W and K cycles.

Includes a block version of multigrid. Most effective for using as a preconditioner for Block Krylov methods (see KrylovMethods.jl).
Coarsest Grid can be solved using [`MUMPS`] (https://github.com/JuliaSparse/MUMPS.jl) or using Julia's backslash.

# Requirements

This package is intended to use with julia versions 0.6.x.

This package is an add-on for [`jInv`](https://github.com/JuliaInv/jInv.jl), which needs to be installed (for regular mesh module).
For the testing -   [`DivSigGrad`] (https://github.com/JuliaInv/DivSigGrad.jl) needs to be installed too.

# Installation

```
Pkg.clone("https://github.com/JuliaInv/jInv.jl","jInv")
Pkg.clone("https://github.com/JuliaInv/DivSigGrad.jl","DivSigGrad")
Pkg.clone("https://github.com/JuliaInv/Multigrid.jl","Multigrid")
Pkg.clone("https://github.com/JuliaInv/ParSpMatVec.jl","ParSpMatVec")
Pkg.build("ParSpMatVec");

Pkg.test("Multigrid")
```

# Examples

Under "examples/SAforDivSigGrad.jl" you can find the 2D experiment that was shown in the paper above. 
