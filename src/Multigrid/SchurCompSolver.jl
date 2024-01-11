export SchurCompSolver,getSchurCompSolver,copySolver,solveLinearSystem,solveLinearSystem!,setupSolver

mutable struct SchurCompSolver<: AbstractSolver
	Mesh
	Sinv
	CT :: SparseMatrixCSC
	Dinv:: SparseMatrixCSC
	B  :: SparseMatrixCSC
	ST :: SparseMatrixCSC
	sym::Int # 0 = unsymmetric, 1 = symmetric s.t A = A';
	isTransposed::Int
	doClear::Int
	facTime::Real
	nSolve::Int
	solveTime::Real
	nFac::Int
end


function getSchurCompSolver(Sinv = [];sym = 0,isTransposed = 0, doClear = 0)
	return SchurCompSolver([],Sinv,spzeros(1),spzeros(1),spzeros(1),spzeros(1),sym,isTransposed,doClear,0.0,0,0.0,0);
end

solveLinearSystem(A,B,param::SchurCompSolver,doTranspose::Int=0) = solveLinearSystem!(A,B,[],param,doTranspose)

function setupSolver(A::SparseMatrixCSC,param::SchurCompSolver)
	tt = time_ns();
	n_cut = size(A,2) - prod(param.Mesh.n);
	B = A[1:n_cut,(n_cut+1):end];
	CT = A[(n_cut+1):end,1:n_cut];
	D = A[(n_cut+1):end,(n_cut+1):end];
	AA = A[1:n_cut,1:n_cut];
	D.nzval .= 1.0./D.nzval;
	S = AA - B*D*CT;
	if isa(param.Sinv, parallelJuliaSolver)
		param.Sinv = setupSolver(S, param.Sinv);
	elseif isa(param.Sinv, hybridKaczmarz)
		param.ST = sparse(S');
		param.Sinv = setupHybridKaczmarz(param.Sinv,param.ST,param.Mesh)
		getHybridKaczmarzPrecond(param.Sinv,param.ST,1)
	else
		println("WARNING: reverting to Julia's LU");
		param.Sinv = lu(S);
	end
	param.CT = CT;
	param.Dinv = D;
	param.B = B;
	param.facTime+= (tt-time_ns())/1e+9; 
	param.nFac+=1
	return param;
end



function solveLinearSystem!(AT::SparseMatrixCSC,B,X,param::SchurCompSolver,doTranspose=0)
	if issparse(B)
		if length(size(B))==1
			B = Vector(B);
		else
			B = Matrix(B);
		end
	end
	if param.doClear == 1
		clear!(param)
	end
	if param.Sinv == []
		println("solveLinearSystem!: See transpose or not here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		param = setupSolver(A,param);
	end
	tt = time_ns()
	n_cut = size(B,1) - prod(param.Mesh.n);
	Q1 = B[1:n_cut];
	Q2 = B[(n_cut+1):end];
	R = (Q1 - param.B*param.Dinv*Q2);
	if isa(param.Sinv, parallelJuliaSolver)
		U1 = solve(R,copy(R),param.Sinv,doTranspose);
	elseif isa(param.Sinv, hybridKaczmarz)
		# U1 = copy(R);
		# U1.= 0.0;
		# applyHybridKaczmarz(param.Sinv,param.ST,R,U1,prod(param.Sinv.numDomains));
		x0 = copy(R); x0.=0.0;
		S = X->param.ST'*X;
		U1, flag,rnorm,iter = KrylovMethods.fgmres(S,R,5,tol = 0.1,maxIter = 5,M = param.Sinv.precond,out=-1,x = x0,flexible=true);
		# println("Relative err coarsest: ",norm(param.ST'*U1 - R)/norm(R))
	else
		U1 = param.Sinv\R;
	end
	U2 = param.Dinv*(Q2 - param.CT*U1);
	U = [U1;U2];
	param.solveTime+=(tt-time_ns())/1e+9; 
	param.nSolve+=1
	return U, param
end # function solveLinearSystem

function clear!(param::SchurCompSolver)
	param.Sinv = [];
	param.isTransposed = 0;
	param.doClear = 0;
	error("Implement this");
end

function copySolver(Ainv::SchurCompSolver)
	return getSchurCompSolver(Ainv.Sinv);
end
