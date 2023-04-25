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
using Multigrid.ParallelJuliaSolver
using Distributed
using KrylovMethods

const DDIndType = UInt32;

export DomainDecompositionOperatorConstructor,DomainDecompositionParam,DomainDecompositionPreconditionerParam,getDomainDecompositionParam,getDDpreconditioner


mutable struct DomainDecompositionPreconditionerParam{VAL,IND}
	sub_problem_param
	i				:: Array{Int64,1}
	A_i				:: SparseMatrixCSC{VAL,IND}
	DirichletMass	:: Array{VAL}
	Ainv			:: AbstractSolver
end

mutable struct DomainDecompositionParam{VAL,IND} <: AbstractSolver
	PrecParams 			::Union{Array{DomainDecompositionPreconditionerParam{VAL,IND},1},Array{RemoteChannel,1}}
	GlobalIndices		::Array{Array{DDIndType}}
	Mesh				::RegularMesh
	numDomains			::Array{Int64,1}
	overlap   			::Array{Int64,1}
	getIndicesOfCell	::Function
	Ainv				::AbstractSolver
	getSubDomainMass	::Function
	workers				::Array{Int64}
	constructor			::Any
	out					::Int64
	doClear				::Int
	nFac				::Int
	facTime				::Real
	nSolve				::Int
	solveTime			::Real
end
function getDomainDecompositionParam(VAL::Type,IND::Type,Mesh,numDomains,overlap,getIndicesOfCell,Ainv::AbstractSolver,getMass::Function = identity,workers::Array{Int64} = (Int64)[])
	return DomainDecompositionParam{VAL,IND}((DomainDecompositionPreconditionerParam{VAL,IND})[],[],Mesh,numDomains,overlap,getIndicesOfCell,Ainv,getMass,workers,getEmptyCtor(VAL,IND),0,0,0,0.0,0,0.0);
end
mutable struct DomainDecompositionOperatorConstructor{VAL,IND}
	problem_param	::Any
	getSubParams  	::Function
	getOperator		::Function
	getDirichletMass::Function
end



function getEmptyCtor(VAL::Type,IND::Type)
	return DomainDecompositionOperatorConstructor{VAL,IND}(0,identity,identity,identity);
end

include("DDIndices.jl");
include("DDService.jl")
include("DDSerial.jl");
include("DDParallel.jl");



import Base.isempty
function isempty(p::DomainDecompositionParam{VAL,IND}) where {VAL,IND}
	return isempty(p.PrecParams);
end

import jInv.LinearSolvers.copySolver;
function copySolver(s::DomainDecompositionParam{VAL,IND}) where {VAL,IND}
	# copies absolutely what's necessary.
	getDomainDecompositionParam(VAL,IND,s.Mesh,s.numDomains,s.overlap,s.getIndicesOfCell,copySolver(s.Ainv),s.getSubDomainMass,s.workers);
end


# import jInvUtils.clear!
function clear!(s::DomainDecompositionParam{VAL,IND}) where {VAL,IND}
	for k=1:length(s.DDPreconditioners)
		clear!(s.DDPreconditioners[k]);
	end
	s.doClear = 0;
end


import jInv.LinearSolvers.setupSolver
function setupSolver(A::SparseMatrixCSC,DDparam::DomainDecompositionParam{VAL,IND}) where {VAL,IND}
	return setupDDSerial(sparse(A'),DDparam);
end




import jInv.LinearSolvers.solveLinearSystem!;
function solveLinearSystem!(At,B,X,param::DomainDecompositionParam{VAL,IND},doTranspose=0) where {VAL,IND}
	
	if param.doClear==1
		clear!(param);
	end
	# build preconditioner
	if isempty(param)
		if length(param.workers) <= 1
			setupDDSerial(At,param);
		else
			setupDDParallel(At,param,param.workers);
		end
	end 
	flag = 0; rnorm = 0.0; iter = 0; resvec = [];
	if !isempty(B)
		if issparse(B)
			B = full(B);
		end
		if size(B,2) == 1
			B = vec(B);
		end
		if norm(B) == 0.0
			X[:] .= 0.0;
			return X, param,flag,rnorm,iter,resvec;
		end
		
		
		# Prec = r->solveGSDDSerial(At,r,zeros(eltype(X),size(X)),param,1,doTranspose)[1];
		Az = zeros(eltype(B),size(B));
		Afun = z->(SpMatMul(At,z,Az,4);return Az;); 
		X,flag,rnorm,iter,resvec = KrylovMethods.fgmres(Afun,B,5,tol = 1e-6,maxIter = 10,M = Prec, x = X,out=2,flexible=true);
		#X,flag,rnorm,iter,resvec = KrylovMethods.fgmres(getAfun(At,zeros(eltype(X),size(X)),4),B,5,tol = 1e-6,maxIter = 100,M = Prec, x = X,out=1,flexible=true);
		# solveDD(At,B,X,param,doTranspose)
	end	
	return X,param,flag,rnorm,iter,resvec
end 

function getDDpreconditioner(At::SparseMatrixCSC, param::DomainDecompositionParam{VAL,IND},B,doTranspose=0) where {VAL,IND}
	x0 = zeros(VAL,size(B));
	x_new = zeros(eltype(B),size(B));
	rt = zeros(VAL,size(B));
	if length(param.workers) <= 1
		Prec = r->(x0[:] .= 0.0; rt[:] .= r; x_new[:].= solveDDSerial(At,rt,x0,param,1,doTranspose)[1]; return x_new);
	else
		Prec = r->(x0[:] .= 0.0; rt[:] .= r; x_new[:].= solveDDParallel(At,rt,x0,param,param.workers,1,doTranspose)[1]; return x_new);
	end
	return Prec
end





end


