export MGsolver,getMGsolver
export solveLinearSystem!,clear!,copySolver, setupSolver

## this is a wrapper for using the SA AMG preconditioner using the abstractSolver interface. 

mutable struct MGsolver{VAL,IND} <: AbstractSolver 
	MG				:: MGparam{VAL,IND}
	Krylov			:: String
	sym				:: Int64  #0=unsymmetric, 1=symm. pos def, 2=general symmetric
	out				:: Int64
	isTranspose 	:: Bool # if true, transpose(A) is provided to solver. If not, A is transposed 
							# note that A_mul_B! is slower than Ac_mul_B for SparseMatrixCSC
	doClear			:: Int64 # flag to clear setup
	tol				:: Float64
	nIter			::Int
	timeSetup		::Real
	timeSolve		::Real
end


function getMGsolver(MG::MGparam{VAL,IND},Mesh::RegularMesh,sym,Krylov::String="GMRES"; out::Int64 =-1) where {VAL,IND}
	MG.Meshes = [Mesh];
	return MGsolver{VAL,IND}(MG,Krylov,sym,out,false,0,MG.relativeTol,0,0.0,0.0);
end

import jInv.LinearSolvers.solveLinearSystem!;
function solveLinearSystem!(A,B::Array{VAL},X::Array{VAL},param::MGsolver{VAL,IND},doTranspose=0) where {VAL,IND}
	if issparse(B)
		B = full(B);
	end
	if size(B,2) == 1
		B = vec(B);
	end
	if param.doClear==1
		clear!(param.MG);
	end
	if norm(B) == 0.0
		X[:] = 0.0;
		return X, param;
	end
	n = size(B,1)
	nrhs = size(B,2);

	# build preconditioner
	if param.out > 0
		verbose = true;
	else
		verbose = false;
	end
	if hierarchyExists(param.MG)==false
		doTransposeIterative = (param.isTranspose) ? mod(doTranspose+1,2) : doTranspose
		if (param.sym==1) ||  ((param.sym != 1) && (doTransposeIterative == 1)) 
			# this means that we're OK with using Ac_mul_B! in iterative methods, hence, MG is also OK.
		elseif (param.sym != 1) && (doTransposeIterative == 0)
			A = sparse(A');
		end
		param.timeSetup += @elapsed MGsetup(A,param.MG.Meshes[1],param.MG,nrhs,verbose);
		param.MG.doTranspose = doTranspose; # the doTranspose of MG MUST be synced with the doTranspose of the interface.
	end 
	
	if (param.sym != 1) && (doTranspose != param.MG.doTranspose)
		param.timeSetup += @elapsed transposeHierarchy(param.MG);
	end
	BLAS.set_num_threads(param.MG.numCores);
	Afun = getAfun(param.MG.As[1],zeros(VAL,size(B)),param.MG.numCores);
	time = time_ns();
	if param.Krylov=="BiCGSTAB"
		X, param.MG,num_iter = solveBiCGSTAB_MG(Afun,param.MG,B,X,verbose);
	elseif param.Krylov=="GMRES"
		X, param.MG,num_iter = solveGMRES_MG(Afun,param.MG,B,X,true,5,verbose);
	elseif param.Krylov=="PCG"
		X, param.MG,num_iter = solveCG_MG(Afun,param.MG,B,X,verbose);
	end
	param.nIter += num_iter*size(X,2);
	param.timeSolve+=(time_ns() - time)/1e+9;

	if num_iter >= param.MG.maxOuterIter - 1
		warn("MG solver reached maximum iterations without convergence");
	end
	return X, param
end 

function setupSolver(AT::SparseMatrixCSC, s::MGsolver{VAL,IND}) where {VAL,IND}
	s.MG = MGsetup(AT,s.MG.Meshes[1],s.MG,1,s.out>0);
	return s;
end

import jInv.LinearSolvers.copySolver
function copySolver(s::MGsolver{VAL,IND}) where {VAL,IND}
	# copies absolutely what's necessary.
	return MGsolver(Multigrid.copySolver(s.MG),copy(s.MG.Meshes[1]),s.sym,s.Krylov,s.out,s.isTranspose,s.doClear,s.tol,0,0.0,0.0);
end


import jInv.Utils.clear!
function clear!(s::MGsolver{VAL,IND}) where {VAL,IND}
	 clear!(s.MG);
	 s.doClear = 0;
end