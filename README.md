[![Build Status](https://travis-ci.org/JuliaInv/Multigrid.jl.svg?branch=master)](https://travis-ci.org/JuliaInv/Multigrid.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaInv/Multigrid.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaInv/Multigrid.jl?branch=master)

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

This package is intended to use with julia versions 0.7-1.0.

This package is an add-on for [`jInv`](https://github.com/JuliaInv/jInv.jl), which needs to be installed (mostly for regular mesh module).

# Installation

```
Pkg.clone("https://github.com/JuliaInv/jInv.jl","jInv")
Pkg.clone("https://github.com/JuliaInv/Multigrid.jl","Multigrid")
Pkg.clone("https://github.com/JuliaInv/ParSpMatVec.jl","ParSpMatVec")
Pkg.build("ParSpMatVec");

Pkg.test("Multigrid")
```
