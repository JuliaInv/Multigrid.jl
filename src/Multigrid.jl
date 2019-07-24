module Multigrid
	using LinearAlgebra
	using SparseArrays
	
	hasParSpMatVec = false
	try
		using ParSpMatVec;
		global hasParSpMatVec = false; #ParSpMatVec.isBuilt();
		if hasParSpMatVec == false
			println("hasParSpMatVec==false. ParSpMatVec has failed to build!!!!!!!");
		end
	catch
		println("ParSpMatVec has failed to load in try-catch!!!!!!!");
	end
	
	import jInv.LinearSolvers.AbstractSolver
	import jInv.LinearSolvers.solveLinearSystem!,jInv.LinearSolvers.solveLinearSystem, jInv.LinearSolvers.setupSolver
	
	SparseCSCTypes = Union{SparseMatrixCSC{ComplexF64,Int64},SparseMatrixCSC{Float64,Int64},SparseMatrixCSC{ComplexF32,Int64},SparseMatrixCSC{Float32,Int64}}
	ArrayTypes = Union{Array{ComplexF64},Array{ComplexF32},Array{Float64},Array{Float32}}
	
	include("DomainDecomposition/DomainDecomposition.jl");
	include("Multigrid/MGdef.jl");
end


# check if MUMPS can be used
# const minMUMPSversion = VersionNumber(0,0,1)
# hasMUMPS=false
# vMUMPS = VersionNumber(0,0,0)
# try 
	# vMUMPS = Pkg.installed("MUMPS")
	# hasMUMPS = vMUMPS >= minMUMPSversion
	# if hasMUMPS
		# using MUMPS;
	# end
# catch 
# end

# end # module
