
export solveDDSerial,setupDDSerial#,getSubMeshOfCell;#,solveDDJac,solveDDRB



function computeResidualAtIdx(AT::SparseMatrixCSC,b::Array,x::Array,Idxs::Array{DDIndType})
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


function performSetup(i::Array{Int64}, AI::SparseMatrixCSC,  DDparam::DomainDecompositionParam)
	# println("setupping sub domain: ",i," by ",myid());
	DDPrec = DomainDecompositionPreconditionerParam([],i,AI,[],copySolver(DDparam.Ainv))
	(~,DDPrec.Ainv) = solveLinearSystem!(AI,zeros(eltype(AI),size(AI,2)),zeros(eltype(AI),size(AI,2)),DDPrec.Ainv);
	# println("setupping sub domain: ",i," all-done by ",myid());
	return DDPrec;
end




function performSetup(i::Array{Int64}, AI::DomainDecompositionOperatorConstructor,  DDparam::DomainDecompositionParam)
	AII = AI.getOperator(AI.problem_param);
	DirichletMass = AI.getDirichletMass(DDparam,i)[:];
	DDPrec = DomainDecompositionPreconditionerParam(AI.problem_param,i,AII,DirichletMass,copySolver(DDparam.Ainv))
	AII_new = sparse((AII + Diagonal(DirichletMass))');
	(~,DDPrec.Ainv) = solveLinearSystem!(AII_new,zeros(eltype(AII),size(AII,2)),zeros(eltype(AII),size(AII,2)),DDPrec.Ainv);
	# println("setipping sub domain: ",i," by ",myid());
	return DDPrec;
end



function computeResidual(precParam::DomainDecompositionPreconditionerParam, x, r)
	# r is initialized by b and overrun here
	r .-= precParam.A_i'*x;
	
	
	# AT = precParam.A_i
	# for t = 1:length(r)
		# for tt = AT.colptr[t]:AT.colptr[t+1]-1
			# r[t] = r[t] - conj(AT.nzval[tt])*x[AT.rowval[tt]];
		# end
	# end
	return r;
end



function setupDDSerial(A::Union{SparseMatrixCSC,DomainDecompositionOperatorConstructor},DDparam::DomainDecompositionParam)

M 			= DDparam.Mesh;
numDomains 	= DDparam.numDomains;
overlap 	= DDparam.overlap;
DDPreconditioners 	= Array{DomainDecompositionPreconditionerParam}(undef,prod(numDomains));
DDparam.GlobalIndices = Array{Array{DDIndType}}(undef,prod(numDomains));
n = M.n;
for ii = 1:prod(numDomains)
	i = cs2loc(ii,numDomains);
	IIp = DDparam.getIndicesOfCell(numDomains,overlap, i,n);
	if isa(A,SparseMatrixCSC)==true
		AI = sparse(A[IIp,IIp]');
		DDPreconditioners[ii] = performSetup(i, AI,  DDparam);
	else
		subparams = A.getSubParams(A.problem_param, M,i,numDomains,overlap);
		AI = DomainDecompositionOperatorConstructor(subparams,A.getSubParams,A.getOperator,A.getDirichletMass);
		DDPreconditioners[ii] = performSetup(i, AI,  DDparam);
	end
	IIp = convert(Array{DDIndType},IIp);
	DDparam.GlobalIndices[ii] = IIp;
end
DDparam.PrecParams = DDPreconditioners;
return DDparam;
end

function solveDDSerial(AT::SparseMatrixCSC,b::Array,x::Array,DDparam::DomainDecompositionParam,niter=1,doTranspose::Int64=0)

ncells = DDparam.Mesh.n;
dim = length(ncells);
numDomains = DDparam.numDomains;
overlap = DDparam.overlap;

for k=1:niter
	for color = 1:2^dim
		for ic = 1:prod(numDomains);
			icloc = cs2loc(ic,numDomains);
			if cellColor(icloc)==color
				#IIp = DDparam.getIndicesOfCell(numDomains,overlap,icloc,ncells);
				IIp = DDparam.GlobalIndices[ic];
				if isa(AT,SparseMatrixCSC)==true
					r = computeResidualAtIdx(AT,b,x,IIp);
				else
					r = computeResidualAtIdx(AT,b,x,IIp);
					# r = computeResidual(DDparam.PrecParams[ic],x[IIp],b[IIp]);					
				end
				(t,DDparam.PrecParams[ic]) = solveSubDomain(r,DDparam.PrecParams[ic],doTranspose)
				x[IIp] = x[IIp] + t;
			end
		end
	end
	if niter > 1
		println(k,": ",norm(b - AT'*x)/norm(b))
	end
end

return x,DDparam;
end


function solveSubDomain(r,prec::DomainDecompositionPreconditionerParam,doTranspose::Int64)
	Os = spzeros(0,0);
	t = zeros(eltype(r),size(r));
	t,prec.Ainv = solveLinearSystem!(prec.A_i,r,t,prec.Ainv,doTranspose);
	return t,prec;
end


# function solveDDJac(AT::SparseMatrixCSC,n::Array{Int64},b::ArrayTypes,x::ArrayTypes,getIndicesOfCell::Function,numDomains::Array{Int64,1},overlap::Array{Int64,1},DDfactors::Array{Any,1},niter = 1,doTranspose::Int64=0)
