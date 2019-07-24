
export solveDDSerial,setupDDSerial#,getSubMeshOfCell;#,solveDDJac,solveDDRB


function setupDDSerial(A::Union{SparseMatrixCSC,DomainDecompositionOperatorConstructor},DDparam::DomainDecompositionParam)

M 			= DDparam.Mesh;
numDomains 	= DDparam.numDomains;
overlap 	= DDparam.overlap;
DDPreconditioners = Array{Any}(prod(numDomains));

n = M.n;
for ii = 1:prod(numDomains)
	i = cs2loc(ii,numDomains);
	if isa(A,SparseMatrixCSC)==true
		IIp = DDparam.getIndicesOfCell(numDomains,overlap, i,n);
		AI = A[IIp,IIp]';
	else
		subparams = A.getSubParams(A.param, M,i,numDomains,overlap);
		AI = A.getOperator(subparams);
	end

	DDPreconditioners[ii] = copySolver(DDparam.Ainv);
	### Effectively we perform individual solver setups here.
	(~,DDPreconditioners[ii]) = solveLinearSystem!(AI,[],[],DDPreconditioners[ii]);
end
DDparam.DDPreconditioners = DDPreconditioners;
return DDPreconditioners;
end

function solveDD(AT::SparseMatrixCSC,b::ArrayTypes,x::ArrayTypes,DDparam::DomainDecompositionParam,doTranspose::Int64=0)

ncells = DDparam.Mesh.n;
niter = 1;

#TODO: See if n divides;
dim = length(ncells);
numDomains = DDparam.numDomains;
overlap = DDparam.overlap;

Ai = spzeros(0,0);
for k=1:niter
	for color = 1:2^dim
		for ic = 1:prod(numDomains);
			icloc = cs2loc(ic,numDomains);
			if cellColor(icloc)==color
				IIp = DDparam.getIndicesOfCell(numDomains,overlap,icloc,ncells);
				r = computeResidualAtIdx(AT,b,x,IIp);
				t = zeros(eltype(r),size(r));
				(t,DDparam.DDPreconditioners[ic]) = solveLinearSystem!(Ai,r,t,DDparam.DDPreconditioners[ic],doTranspose);
				x[IIp] = x[IIp] + t;
			end
		end
	end
	# if niter >= 1
		# println(k,": ",norm(b - AT'*x)/norm(b))
	# end
end

return x;
end


# function solveDDJac(AT::SparseMatrixCSC,n::Array{Int64},b::ArrayTypes,x::ArrayTypes,getIndicesOfCell::Function,numDomains::Array{Int64,1},overlap::Array{Int64,1},DDfactors::Array{Any,1},niter = 1,doTranspose::Int64=0)
# ncells = n;
# #TODO: See if n divides;
# dim = length(n);
# z = real(copy(x));
# for k=1:niter
	# r = b - AT'*x;
	# y = copy(x);
	# z[:] = 0.0
	# for ic = 1:prod(numDomains);
		# icloc = cs2loc(ic,numDomains);
		# IIp,Wp = getIndicesOfCell(numDomains,overlap,icloc,ncells);
		# rI = r[IIp];
		# t = applyMUMPS(DDfactors[ic],rI,rI*0.0,doTranspose);
		# x[IIp] = y[IIp] + Wp.*t;
		# z[IIp] = z[IIp] + Wp;
		# # if sum(Wp.>1.0 + 1e-15) > 0
			# # println(Wp)
			# # error("Im here")
		# # end
		# # if sum(z .> 1.0 + 1e-15) > 0
			# # println(maximum(z))
			# # println(z)
			# # error("Im here 2")
		# # end
	# end
	# # if niter >= 1
		# # println(k,": ",norm(b - AT'*x)/norm(b))
	# # end
	# if sum(z.!=1.0) > 0
		# println(maximum(z))
		# println(minimum(z))
		# error("Im here")
	# end
# end

# return x;
# end


# function solveDDRB(AT::SparseMatrixCSC,n::Array{Int64},b::ArrayTypes,x::ArrayTypes,getIndicesOfCell::Function,numDomains::Array{Int64,1},overlap::Array{Int64,1},DDfactors::Array{Any,1},niter = 1,doTranspose::Int64=0)
# ncells = n;
# #TODO: See if n divides;
# dim = length(n);
# z = copy(x);
# for k=1:niter
	# z[:] = 0.0;
	# for color = 1:2
		# r = b - AT'*x;
		# y = copy(x);
		# for ic = 1:prod(numDomains);
			# icloc = cs2loc(ic,numDomains);
			# if cellRBColor(icloc)==color
				# IIp,Wp = getIndicesOfCell(numDomains,overlap,icloc,ncells);
				# # r = computeResidualAtIdx(AT,b,x,IIp);
				# # t = solve(r,DDfactors[ic]);
				# # t = DDfactors[ic]\r;
				# rI = r[IIp];
				# t = applyMUMPS(DDfactors[ic],rI,rI*0.0,doTranspose);
				# x[IIp] = y[IIp] + Wp.*t;
				# z[IIp] = z[IIp] + Wp;
			# end
		# end
	# end
	# # if niter >= 1
		# # println(k,": ",norm(b - AT'*x)/norm(b))
	# # end
	# if sum(z.!=1.0) > 0
		# println(z)
		# error("Im here")
	# end
# end
# return x;
# end




function computeResidualAtIdx(AT::SparseMatrixCSC,b::Array,x::Array,Idxs::Array{Int64})
r = b[Idxs];
for t = 1:length(Idxs)
	for tt = AT.colptr[Idxs[t]]:AT.colptr[Idxs[t]+1]-1
		r[t] = r[t] - conj(AT.nzval[tt])*x[AT.rowval[tt]];
	end
end

#void computeResidualAtIdx(long long *rowptr , double complex *valA ,long long *colA,double complex *b,double complex *x,long long* Idxs, double complex *local_r, int blockSize){
# local_r = r*0.0;
# ccall((:computeResidualAtIdx,lib),Void,(Ptr{Int64},Ptr{Complex128},Ptr{Int64},Ptr{Complex128},Ptr{Complex128},Ptr{Int64},Ptr{Complex128},Int16,)
								 # ,AT.colptr,AT.nzval,AT.rowval,b,x,Idxs,local_r,convert(Int16,length(Idxs)));
# if norm(local_r - r) > 1e-14
	# error("Fix computeResidualAtIdx in C: ",norm(local_r - r));
# end
return r;
end











