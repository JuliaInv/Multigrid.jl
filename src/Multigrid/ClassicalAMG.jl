export ClassicalAMGsetup;
include("coloring.jl")
include("interpolation.jl")

function ClassicalAMGsetup(AT::SparseMatrixCSC{VAL,IND},param::MGparam{VAL,IND},
							symm::Bool = true,nrhs::Int64 = 1,verbose::Bool=false) where {VAL,IND}


Ps = Array{SparseMatrixCSC{real(VAL),IND}}(undef, param.levels-1);
Rs = Array{SparseMatrixCSC{real(VAL),IND}}(undef, param.levels-1);
As = Array{SparseMatrixCSC{VAL,IND}}(undef, param.levels);

if symm == false
	error("not supported yet...");
end

N = size(AT,2);
As[1] = AT;
relaxPrecs = Array{Vector{VAL}}(undef, param.levels-1);
Cop = nnz(AT);
time = 0;
for l = 1:(param.levels-1)
	if verbose
		time = time_ns();
	end
	AT = As[l];
	if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES" || param.relaxType=="SPAI"
		relaxPrecs[l] = getRelaxPrec(AT,param.relaxType,param.relaxParam,[],false);
	else
		error("Unknown relaxation type !!!!");
	end
	
	
	
	S = getStrengthMatrix(AT,param.strongConnParam, N);
	#println("Time of first coloring pass:")
	coloring = getColoringFirst(S, N)
	checkColoring(coloring,N)
	#println("Time of second coloring pass:")
	coloring = getColoringSecond(S, coloring, N)
	checkColoring(coloring,N)
	#println("Time of interpolation:")
	(P, PT) = getInterpolation(AT, S, coloring, N)
	Nc = size(P,2);
	## P should also be row-wise, so we hold P^T.
	if (size(P,1)==size(P,2))
		if verbose; println(string("Stopped Coarsening at level ",l)); end
		param.levels = l;
		As = As[1:l];
		Ps = Ps[1:l-1];
		Rs = Rs[1:l-1];
		relaxPrecs = relaxPrecs[1:l-1];
		break;
	else
		Rs[l] = P; # this is becasue we hold the transpose of the matrices and P = R' anyway here....
		Ps[l] = PT;

		Act = Ps[l]*AT*Rs[l];
		As[l+1] = Act;
		Cop = Cop + nnz(Act);

		if verbose; println("MG setup: ",N," took:", (time_ns()-time)/1e+9); end;
		N = Nc;
	end
end
if verbose
	time = time_ns();
	println("MG Setup: Operator complexity = ",Cop/nnz(As[1]));
end
As[end] = As[end] + 1e-8*norm(As[end],1)*sparse(1.0I,size(As[end],2),size(As[end],2));
defineCoarsestAinv(param,As[end]);

if verbose
	println("MG setup coarsest: ",N,", done LU in ",(time_ns()-time)/1e+9);
end
param.As = As;
param.Ps = Ps;
param.Rs = Rs;
param.relaxPrecs = relaxPrecs;
param = adjustMemoryForNumRHS(param,nrhs,verbose);
return;
end

function getStrengthMatrix(AT::SparseMatrixCSC,strengthConnParam::Float64, n::Int64)
S = -AT;
mm = 1e-16*maximum(S.nzval);
for j = 1:n
	maxVal_j = mm;
	for gidx = S.colptr[j] : S.colptr[j+1]-1 ### here we go over the j-th row
		if S.nzval[gidx] > maxVal_j
			maxVal_j = S.nzval[gidx];
		end
	end
	scal_k = 1.0./maxVal_j;
	for gidx = S.colptr[j] : S.colptr[j+1]-1
		S.nzval[gidx]*=scal_k;
	end
	for gidx = S.colptr[j] : S.colptr[j+1]-1
		if S.nzval[gidx] <  strengthConnParam
			S.nzval[gidx] = 0.0;
		end
	end
	for gidx = S.colptr[j] : S.colptr[j+1]-1
		if S.rowval[gidx] == j
			# We have a diagonal element.
			S.nzval[gidx] = 1.0;
		end
	end
end
S = (S + S')/2;
dropzeros!(S)
end



function checkColoring(coloring::Array, n::Int64)
	coarse = 0
	fine = 0
	for i=1:n
		if coloring[i] == 1
			coarse+=1
		else
			fine+=1
		end
	end
	#println("fine : $fine , coarse : $coarse")
end







