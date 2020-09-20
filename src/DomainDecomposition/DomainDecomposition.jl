module DomainDecomposition
using SparseArrays
using LinearAlgebra
using jInv.Mesh
using jInv.Utils
import jInv.LinearSolvers.AbstractSolver
import jInv.LinearSolvers.solveLinearSystem
import jInv.LinearSolvers.solveLinearSystem!
import jInv.LinearSolvers.copySolver;
using Multigrid
using Distributed
using KrylovMethods

const DDIndType = UInt32;

export DomainDecompositionOperatorConstructor,DomainDecompositionParam,DomainDecompositionPreconditionerParam,getDomainDecompositionParam


mutable struct DomainDecompositionPreconditionerParam
	sub_problem_param
	i				:: Array{Int64,1}
	A_i				:: SparseMatrixCSC
	DirichletMass	:: Vector
	Ainv			:: AbstractSolver
end

mutable struct DomainDecompositionParam <: AbstractSolver
	PrecParams 			::Union{Array{DomainDecompositionPreconditionerParam,1},Array{RemoteChannel,1}}
	GlobalIndices		::Array{Array{DDIndType}}
	Mesh				::RegularMesh
	numDomains			::Array{Int64,1}
	overlap   			::Array{Int64,1}
	getIndicesOfCell	::Function
	Ainv				::AbstractSolver
	workers				::Array{Int64}
	constructor			::Any
	out					::Int64
	doClear				::Int
	nFac				::Int
	facTime				::Real
	nSolve				::Int
	solveTime			::Real
end
function getDomainDecompositionParam(Mesh,numDomains,overlap,getIndicesOfCell,Ainv::AbstractSolver)
	return DomainDecompositionParam((DomainDecompositionPreconditionerParam)[],[],Mesh,numDomains,overlap,getIndicesOfCell,Ainv,(Int64)[],getEmptyCtor(),0,0,0,0.0,0,0.0);
end
mutable struct DomainDecompositionOperatorConstructor
	problem_param	::Any
	getSubParams  	::Function
	getOperator		::Function
	getDirichletMass::Function
end



function getEmptyCtor()
	return DomainDecompositionOperatorConstructor(0,identity,identity,identity);
end

include("DDIndices.jl");
include("DDService.jl")
include("DDSerial.jl");
include("DDParallel.jl");



import Base.isempty
function isempty(p::DomainDecompositionParam)
	return isempty(p.PrecParams);
end

import jInv.LinearSolvers.copySolver;
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
		if norm(B) == 0.0
			X[:] .= 0.0;
			return X, param;
		end
		Prec = r->solveDDSerial(At,r,zeros(eltype(X),size(X)),param,1,doTranspose)[1];
		x, flag,rnorm,iter = KrylovMethods.fgmres(getAfun(At,zeros(eltype(X),size(X)),4),B,10,tol = 1e-10,maxIter = 5,M = Prec, x = X,out=2,flexible=true);
		
		# solveDD(At,B,X,param,doTranspose)
	end	
	return X, param
end 






end


