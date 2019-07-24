module DomainDecomposition
using SparseArrays
using LinearAlgebra
using jInv.Mesh
import jInv.LinearSolvers.AbstractSolver
import Multigrid.ArrayTypes
import jInv.LinearSolvers.solveLinearSystem
import jInv.LinearSolvers.solveLinearSystem!
import jInv.LinearSolvers.copySolver;
using KrylovMethods



export DomainDecompositionOperatorConstructor,DomainDecompositionParam,getDomainDecompositionParam
mutable struct DomainDecompositionParam <: AbstractSolver
	DDPreconditioners 	::Array{AbstractSolver,1}
	Mesh				::RegularMesh
	numDomains			::Array{Int64,1}
	overlap   			::Array{Int64,1}
	getIndicesOfCell	::Function
	Ainv				::AbstractSolver
	workers				::Array{Int64}
	constructor
	out::Int64
	doClear::Int
	nFac::Int
	facTime::Real
	nSolve::Int
	solveTime::Real
end
function getDomainDecompositionParam(Mesh,numDomains,overlap,getIndicesOfCell,Ainv::AbstractSolver)
	return DomainDecompositionParam((AbstractSolver)[],Mesh,numDomains,overlap,getIndicesOfCell,Ainv,(Int64)[],getEmptyCtor(),0,0,0,0.0,0,0.0);
end
mutable struct DomainDecompositionOperatorConstructor
	problem_param
	getSubParams  	::Function
	getOperator		::Function
end
function getEmptyCtor()
	return DomainDecompositionOperatorConstructor(0,identity,identity);
end

include("DDSerial.jl");
include("DDIndices.jl");
include("parallelJuliaSolver.jl");
include("DDService.jl")


import Base.isempty
function isempty(p::DomainDecompositionParam)
	return isempty(p.DDPreconditioners);
end


function copySolver(s::DomainDecompositionParam)
	# copies absolutely what's necessary.
	error("TODO");
end


# import jInvUtils.clear!
function clear!(s::DomainDecompositionParam)
	for k=1:length(s.DDPreconditioners)
		clear!(s.DDPreconditioners[k]);
	end
	s.doClear = 0;
end


import jInv.LinearSolvers.solveLinearSystem!;
function solveLinearSystem!(At,B,X,param::DomainDecompositionParam,doTranspose=0)
	
	if param.doClear==1
		clear!(param);
	end
	# build preconditioner
	if isempty(param)
		setupDDSerial(At,param);
	end 

	
	
	if !isempty(B)
		if issparse(B)
			B = full(B);
		end
		if size(B,2) == 1
			B = vec(B);
		end
		if vecnorm(B) == 0.0
			X[:] = 0.0;
			return X, param;
		end
		Prec = r->solveDD(At,r,zeros(eltype(X),size(X)),param,doTranspose);
		x, flag,rnorm,iter = KrylovMethods.fgmres(getAfun(At,zeros(eltype(X),size(X)),4),B,4,tol = 0.01,maxIter = 1,M = Prec, x = X,out=0,flexible=true);
		
		# solveDD(At,B,X,param,doTranspose)
	end	
	return X, param
end 






end


