#import jInvLinearSolvers.AbstractSolver

mutable struct parallelJuliaSolver <: AbstractSolver
	L::SparseMatrixCSC;
	U::SparseMatrixCSC;
	p::Union{Array{Int32},Array{Int64}};
	q::Union{Array{Int32},Array{Int64}};
	numCores::Int64
	compress::Int64
	doClear::Int
	nFac::Int
	facTime::Real
	nSolve::Int
	solveTime::Real
end
export getParallelJuliaSolver;
function  getParallelJuliaSolver(numCores=1)
	return parallelJuliaSolver(spzeros(0),spzeros(0),(Int64)[],(Int64)[],numCores,0,0,0,0.0,0,0.0);
end

#import jInvLinearSolvers.solveLinearSystem
solveLinearSystem(A,B,param::parallelJuliaSolver,doTranspose::Int=0) = solveLinearSystem!(A,B,copy(B),param,doTranspose)

#import jInvLinearSolvers.solveLinearSystem!
function solveLinearSystem!(A::SparseMatrixCSC,B,X,param::parallelJuliaSolver,doTranspose=0)
	if doTranspose==1
		error("TODO:handle this");
	end
	
	if param.doClear == 1
		clear!(param)
	end
	if isempty(param.L)
		tic()
		(param.L,param.U,param.p,param.q) = setupLUFactor(A);
		param.facTime+=toq()
		param.nFac+=1
	end
	if !isempty(B)
		tic()
		X = solve(B,X,param,doTranspose);
		param.solveTime+=toq()
		param.nSolve+=1
	end
	

	return X, param
end # function solveLinearSystem

function getOps(LU::parallelJuliaSolver)
	return (LU.L,LU.U,LU.p,LU.q);
end

function setupLUFactor(AI::SparseMatrixCSC,compress = 0)
	# return factorMUMPS(AI,0,0);
	LU = lufact(AI);
	L,U,p,q,Rs = LU[:(:)];
	LU = 0;

	# if dropTol > 0.0
		# L.nzval[abs(L.nzval).<=dropTol] = 0.0;
		# L = L + spzeros(L.m,L.n);
		# d = 1./diag(U);
		# U = d.*U;
		# U.nzval[abs(U.nzval).<=dropTol] = 0.0;
		# U = U + spzeros(U.m,U.n);
		# U = (1./d).*U;
	# end
	Rs = 1.0./Rs[p];
	for i=1:length(L.nzval)
		@inbounds L.nzval[i]*=Rs[L.rowval[i]];
	end
	
	# @time L = (1./Rs[p]).*L;
	# @time L = SparseMatrixCSC(size(L,1),size(L,2),convert(Array{Int32},L.colptr),convert(Array{Int32},L.rowval),convert(Array{Complex32},L.nzval));
	L = convert(SparseMatrixCSC{ComplexF32,Int32},L);
	U = convert(SparseMatrixCSC{ComplexF32,Int32},U);
	
	# LU = getCompressedLUfactor(L,U, convert(Array{UInt32},p),convert(Array{UInt32},q));
	return L,U,p,q;
end


function solve(b,x,LU::parallelJuliaSolver,doTranspose=0);
	if length(size(b))==1
		x[LU.q] = (LU.U\(LU.L\(b[LU.p])));
	else
		x[LU.q,:] = (LU.U\(LU.L\(b[LU.p,:])));
	end
	return x;
end
#import jInvLinearSolvers.clear!
function clear!(param::parallelJuliaSolver)
	param.L = spzeros(0,0);
	param.U = spzeros(0,0);
	param.q = (Int64)[];
	param.p = (Int64)[];
	param.doClear = 0;
end

#import jInvLinearSolvers.copySolver
function copySolver(Ainv::parallelJuliaSolver)
	param = getParallelJuliaSolver(Ainv.numCores);
	param.doClear = Ainv.doClear;	
	return param;
end

# function getCompressedLUfactor(L,U, p,q)
	# return LUfactor(L,U,p,q);
# end
# type LUfactor
	# Lnzval::Array{UInt8}
	# Lrowval::Array{UInt8}
	# Lcolptr::Array{UInt8}
	# Unzval::Array{UInt8}
	# Urowval::Array{UInt8}
	# Ucolptr::Array{UInt8}
	# pcomp::Array{UInt8}
	# qcomp::Array{UInt8}
# end

# using Blosc;
# function getCompressedLUfactor(L,U, p,q)
	# Blosc.set_num_threads(4);
	# Lnzval = compress(L.nzval, level=9, shuffle=true, itemsize=sizeof(Complex32));
	# println(length(Lnzval) / (4*length(L.nzval)))
	# Lrowval = compress(L.rowval, level=9, shuffle=true, itemsize=sizeof(UInt32));
	# println(length(Lrowval) / (4*length(L.rowval)))
	# Lcolptr = compress(L.colptr, level=9, shuffle=true, itemsize=sizeof(UInt32));
	# println(length(Lcolptr) / (4*length(L.colptr)))
	# Unzval = compress(U.nzval, level=9, shuffle=true, itemsize=sizeof(Complex32));
	# Urowval = compress(U.rowval, level=9, shuffle=true, itemsize=sizeof(UInt32));
	# Ucolptr = compress(U.colptr, level=9, shuffle=true, itemsize=sizeof(UInt32));
	# pcomp = compress(p, level=9, shuffle=true, itemsize=sizeof(UInt32));
	# println(length(pcomp) / (4*length(p)))
	# qcomp = compress(q, level=9, shuffle=true, itemsize=sizeof(UInt32));
	# error("ET")
	# return LUfactor(Lnzval,Lrowval,Lcolptr,Unzval,Urowval,Ucolptr,pcomp,qcomp);
# end

# function getOps(LU::LUfactor)
	# Blosc.set_num_threads(4);
	# p = decompress(UInt32, LU.pcomp);
	# q = decompress(UInt32, LU.qcomp);
	# L = SparseMatrixCSC(length(p),length(p),decompress(UInt32, LU.Lcolptr),decompress(UInt32, LU.Lrowval),decompress(Complex32, LU.Lnzval));
	# U = SparseMatrixCSC(length(p),length(p),decompress(UInt32, LU.Ucolptr),decompress(UInt32, LU.Urowval),decompress(Complex32, LU.Unzval));
	# return (L,U,p,q);
# end
