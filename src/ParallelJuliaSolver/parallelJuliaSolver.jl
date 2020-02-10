module ParallelJuliaSolver
	using SparseArrays
	using LinearAlgebra
	import jInv.LinearSolvers.AbstractSolver
	println(splitdir(Base.source_path())[1])
	const parLU_lib  = abspath(joinpath(splitdir(Base.source_path())[1],"../..","deps","builds","parLU"))

mutable struct MySparseMatrixCSR{VAL,IND} 
	nzval::Array{VAL};
	rowptr::Array{Int64};
	colval::Array{IND};
	m::Int64;
	n::Int64;
end

export getEmptySparseMatrixCSR
function getEmptySparseMatrixCSR(VAL::Type,IND::Type,m,n,nnz)
	return MySparseMatrixCSR{VAL,IND}(zeros(VAL,nnz),zeros(Int64,m+1),zeros(IND,nnz),m,n);
end


export convertCSC2MyCSR
function convertCSC2MyCSR(A::SparseMatrixCSC,MyVAL::Type,MyIND::Type)
	AT = sparse(transpose(A)); # this does not perform conjugate	
	return MySparseMatrixCSR{MyVAL,MyIND}(convert(Array{MyVAL},AT.nzval),
	                                   convert(Array{Int64},AT.colptr),
									   convert(Array{MyIND},AT.rowval), AT.m,AT.n);
end

import Base.copy
function copy(A::MySparseMatrixCSR{VAL,IND}) where {VAL,IND}
	B = getEmptySparseMatrixCSR(VAL,IND,A.m,A.n,length(A.nzval));
	B.nzval.=A.nzval;
	B.rowptr.=A.rowptr;
	B.colval.=A.colval;
	return B;
end

import Base.isempty
function isempty(A::MySparseMatrixCSR{VAL,IND}) where {VAL,IND}
	return length(A.nzval)==0;
end


mutable struct parallelJuliaSolver{VAL,IND} <: AbstractSolver
	L::Union{SparseMatrixCSC{VAL,IND},MySparseMatrixCSR{VAL,IND}};
	U::Union{SparseMatrixCSC{VAL,IND},MySparseMatrixCSR{VAL,IND}};
	p::Array{IND};
	q::Array{IND};
	numCores::Int64
	backend::Int64 # 1 for native julia, 2 for CSR julia, 3 for CSR C++
	doClear::Int
	nFac::Int
	facTime::Real
	nSolve::Int
	solveTime::Real
end


export getParallelJuliaSolver;
function getParallelJuliaSolver(VAL::Type,IND::Type;numCores=1,backend=1)
	if backend == 1
		return parallelJuliaSolver{VAL,IND}(spzeros(VAL,IND,0,0),spzeros(VAL,IND,0,0),(IND)[],(IND)[],numCores,backend,0,0,0.0,0,0.0);
	else
		return parallelJuliaSolver{VAL,IND}(getEmptySparseMatrixCSR(VAL,IND,0,0,0),getEmptySparseMatrixCSR(VAL,IND,0,0,0),(IND)[],(IND)[],numCores,backend,0,0,0.0,0,0.0);
	end
end




import jInv.LinearSolvers.solveLinearSystem
function solveLinearSystem(A::SparseMatrixCSC,B::Array{VAL},param::parallelJuliaSolver{VAL,IND},doTranspose::Int64=0) where {VAL,IND}
	return solveLinearSystem!(A,B,copy(B),param,doTranspose);
end

import jInv.LinearSolvers.solveLinearSystem!
function solveLinearSystem!(A::SparseMatrixCSC,B::Array{VAL},X::Array{VAL},param::parallelJuliaSolver{VAL,IND},doTranspose::Int64=0) where {VAL,IND}
	if param.doClear == 1
		clear!(param)
	end
	if isempty(param.L)
		param.facTime += @elapsed param = setupLUFactor(A,param);
		param.nFac+=1
	end
	if !isempty(B)
		param.solveTime += @elapsed X = solve(B,X,param,doTranspose);
		param.nSolve+=1
	end
	return X, param
end # function solveLinearSystem



function setupLUFactor(AI::SparseMatrixCSC,param::parallelJuliaSolver{VAL,IND}) where {VAL,IND}
	LU = lu(AI);
	L = LU.L; U = LU.U; p = LU.p; q = LU.q; Rs = LU.Rs;
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
	
	if param.backend==1
		param.L = convert(SparseMatrixCSC{VAL,IND},L);
		param.U = convert(SparseMatrixCSC{VAL,IND},U);
		param.p = convert(Array{IND},p);
		param.q = convert(Array{IND},q);
	else
		param.L = convertCSC2MyCSR(L,VAL,IND);
		param.U = convertCSC2MyCSR(U,VAL,IND);
		param.p = convert(Array{IND},p);
		param.q = convert(Array{IND},q);
	end
	return param;
end


function solve(b::Array{VAL},x::Array{VAL},LU::parallelJuliaSolver{VAL,IND},doTranspose=0) where {VAL,IND}
	if LU.backend==1
		if doTranspose==0
			if length(size(b))==1
				x[LU.q] = (LU.U\(LU.L\(b[LU.p])));
			else
				x[LU.q,:] = (LU.U\(LU.L\(b[LU.p,:])));
			end
		else
			if length(size(b))==1
				x[LU.p] = ((LU.L')\((LU.U')\(b[LU.q])));
			else
				x[LU.p,:] = ((LU.L')\((LU.U')\(b[LU.q,:])));
			end
		end
	elseif LU.backend==2
		b = copy(b);
		n = size(b,1);
		m = size(b,2);
		L = LU.L; U = LU.U; Lnzval = L.nzval; Lrowptr = L.rowptr; Lcolval=L.colval; p=LU.p; q=LU.q;
		Unzval = U.nzval; Urowptr = U.rowptr; Ucolval=U.colval;
		offset = 0;
		for k=1:m
			for row = 1:n
				@inbounds x[row + offset] = b[p[row]+offset];
			end
			# FORWARD SUBSTITUTION:
			for row = 1:n
				inner_prod = x[row + offset]; #
				for gIdx = Lrowptr[row]:(Lrowptr[row+1]-2) # only until the second to last
					inner_prod -= (Lnzval[gIdx])*x[Lcolval[gIdx]+offset];
				end
				# when transposed, and hence held in CSR, the diagonal elements of L are last.
				diagElem = Lnzval[Lrowptr[row+1]-1]; 
				x[row+offset] = inner_prod/diagElem;
			end
			# BACKWARD SUBSTITUTION:
			for row = n:-1:1
				# when transposed, and hence held in CSR, the diagonal elements of U are first.
				@inbounds diagElem = Unzval[Urowptr[row]];
				@inbounds inner_prod = x[row+offset];
				@inbounds for gIdx = (Urowptr[row]+1):(Urowptr[row+1]-1) # starting from the second element
					@inbounds inner_prod -= Unzval[gIdx]*b[Ucolval[gIdx]+offset];
				          end
				@inbounds b[row+offset] = inner_prod/diagElem;
			end 
			for row = 1:n
				@inbounds x[q[row]+offset] = b[row+offset];
			end
			offset+=n;
		end
	else
		b = copy(b);
		n = size(b,1)*ones(Int64,size(b,2));
		nnz = [length(LU.L.nzval)];
		applyLUSolve(b,x,LU,n,nnz,doTranspose=doTranspose);
	end
	return x;
end

# ccall((:applyLUsolve,parLU_lib),Nothing,(Ptr{Int64},Ptr{VAL},Ptr{IND},Ptr{Int64},Ptr{VAL},Ptr{IND},Ptr{IND},Ptr{IND},Ptr{Int64},Ptr{Int64},Ptr{VAL},Ptr{VAL},Int64, Int64, Int64,Int64,),
			# LU.L.rowptr,LU.L.nzval,LU.L.colval,LU.U.rowptr,LU.U.nzval,LU.U.colval, LU.p,LU.q, n,nnz,x,b,1,size(b,2), LU.numCores,1);
			
function applyLUSolve(b::Array{Float64},x::Array{Float64},LU::parallelJuliaSolver{Float64,Int64},n,nnz;doTranspose=0)
	ccall((:applyLUsolve_FP64_INT64,parLU_lib),Nothing,(Ptr{Int64},Ptr{Float64},Ptr{Int64},Ptr{Int64},Ptr{Float64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Float64},Ptr{Float64},Int64, Int64, Int64,Int64,Int64,),
			LU.L.rowptr,LU.L.nzval,LU.L.colval,LU.U.rowptr,LU.U.nzval,LU.U.colval, LU.p,LU.q, n,nnz,x,b,1,size(b,2), LU.numCores,1,doTranspose);
end

function applyLUSolve(b::Array{Float32},x::Array{Float32},LU::parallelJuliaSolver{Float32,UInt32},n,nnz;doTranspose=0)
	ccall((:applyLUsolve_FP32_UINT32,parLU_lib),Nothing,(Ptr{Int64},Ptr{Float32},Ptr{UInt32},Ptr{Int64},Ptr{Float32},Ptr{UInt32},Ptr{Int32},Ptr{Int32},Ptr{Int64},Ptr{Int64},Ptr{Float32},Ptr{Float32},Int64, Int64, Int64,Int64,Int64,),
			LU.L.rowptr,LU.L.nzval,LU.L.colval,LU.U.rowptr,LU.U.nzval,LU.U.colval, LU.p,LU.q, n,nnz,x,b,1,size(b,2), LU.numCores,1,doTranspose);
end

function applyLUSolve(b::Array{ComplexF64},x::Array{ComplexF64},LU::parallelJuliaSolver{ComplexF64,Int64},n,nnz;doTranspose=0)
	ccall((:applyLUsolve_CFP64_INT64,parLU_lib),Nothing,(Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{ComplexF64},Ptr{ComplexF64},Int64, Int64, Int64,Int64,Int64,),
			LU.L.rowptr,LU.L.nzval,LU.L.colval,LU.U.rowptr,LU.U.nzval,LU.U.colval, LU.p,LU.q, n,nnz,x,b,1,size(b,2), LU.numCores,1,doTranspose);
end


function applyLUSolve(b::Array{ComplexF32},x::Array{ComplexF32},LU::parallelJuliaSolver{ComplexF32,UInt32},n,nnz;doTranspose=0)
	ccall((:applyLUsolve_CFP32_UINT32,parLU_lib),Nothing,(Ptr{Int64},Ptr{ComplexF32},Ptr{UInt32},Ptr{Int64},Ptr{ComplexF32},Ptr{UInt32},Ptr{UInt32},Ptr{UInt32},Ptr{Int64},Ptr{Int64},Ptr{ComplexF32},Ptr{ComplexF32},Int64, Int64, Int64,Int64,Int64,),
			LU.L.rowptr,LU.L.nzval,LU.L.colval,LU.U.rowptr,LU.U.nzval,LU.U.colval, LU.p,LU.q, n,nnz,x,b,1,size(b,2), LU.numCores,1,doTranspose);
end



import jInv.LinearSolvers.clear!
function clear!(param::parallelJuliaSolver{VAL,IND}) where {VAL,IND}
	if param.backend==1
		param.L = spzeros(VAL,IND,0,0);
		param.U = spzeros(VAL,IND,0,0);
	else
		param.L = getEmptySparseMatrixCSR(VAL,IND,0,0,0);
		param.U = getEmptySparseMatrixCSR(VAL,IND,0,0,0);
	end
	param.q = (IND)[];
	param.p = (IND)[];
	param.doClear = 0;
end

import jInv.LinearSolvers.copySolver
function copySolver(Ainv::parallelJuliaSolver{VAL,IND}) where {VAL,IND}
	println("IM HERE!!!")
	param = parallelJuliaSolver{VAL,IND}(copy(Ainv.L),copy(Ainv.U),copy(Ainv.p),copy(Ainv.q),Ainv.numCores,Ainv.backend,0,0,0.0,0,0.0);
	println("IM HERE AGAIN!!!")
	return param;
end

end
