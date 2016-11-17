export SA_AMGsolver,getSA_AMGsolver,copySolver
export solveLinearSystem!,clear!,copySolver

## this is a wrapper for using the SA AMG preconditioner using the abstractSolver interface. 

type SA_AMGsolver <: AbstractSolver
	MG				:: MGparam
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


function getSA_AMGsolver(MG::MGparam,Krylov::String="BiCGSTAB"; sym::Int64=1, out::Int64 =-1)
	if sym!=1
		warning("Non-symmetric AMG version is not implemented yet...")
	end
	return SA_AMGsolver(MG,Krylov,sym,out,false,0,MG.relativeTol,0,0.0,0.0);
end

import jInv.LinearSolvers.solveLinearSystem!;
function solveLinearSystem!(A,B,X,param::SA_AMGsolver,doTranspose=0)
	if issparse(B)
		B = full(B);
	end
	if size(B,2) == 1
		B = vec(B);
	end
	if param.doClear==1
		clear!(param.MG);
	end
	if vecnorm(B) == 0.0
		X[:] = 0.0;
		return X, param;
	end
	TYPE = eltype(B);
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
			A = A';
		end
		tic()
		SA_AMGsetup(A,param.MG,TYPE,param.sym==1,nrhs,verbose);
		param.timeSetup += toq();
		param.MG.doTranspose = doTranspose; # the doTranspose of MG MUST be synced with the doTranspose of the interface.
	end 
	
	if (param.sym != 1) && (doTranspose != param.MG.doTranspose)
		tic()
		transposeHierarchy(param.MG);
		param.timeSetup += toq();
	end
	BLAS.set_num_threads(param.MG.numCores);
	Afun = getAfun(param.MG.As[1],zeros(TYPE,size(B)),param.MG.numCores);
	tic()
	if param.Krylov=="BiCGSTAB"
		X, param.MG,num_iter = solveBiCGSTAB_MG(Afun,param.MG,B,X,verbose);
	elseif param.Krylov=="PCG"
		X, param.MG,num_iter = solveCG_MG(Afun,param.MG,B,X,verbose);
	end
	param.nIter += num_iter*size(X,2);
	param.timeSolve+=toq();

	if num_iter >= param.MG.maxOuterIter - 1
		warn("MG solver reached maximum iterations without convergence");
	end
	return X, param
end 


import jInv.LinearSolvers.copySolver;
function copySolver(s::SA_AMGsolver)
	# copies absolutely what's necessary.
	return SA_AMGsolver(Multigrid.copySolver(s.MG),s.Krylov,s.sym,s.out,s.isTranspose,s.doClear,s.tol,0,0.0,0.0);
end


import jInv.Utils.clear!
function clear!(s::SA_AMGsolver)
	 clear!(s.MG);
	 s.doClear = 0;
end