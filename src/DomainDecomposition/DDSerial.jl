
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
	if DDparam.getSubDomainMass!=identity
		AI = AI + DDparam.getSubDomainMass(DDparam,i);
	end
	DDPrec = DomainDecompositionPreconditionerParam([],i,AI,[],copySolver(DDparam.Ainv))
	DDPrec.Ainv = setupSolver(AI,DDPrec.Ainv);

	if isa(DDPrec.Ainv,ParallelJuliaSolver.parallelJuliaSolver)
		DDPrec.A_i = spzeros(0,0);
	end
	# println("setup for sub domain: ",i," all-done by ",myid());
	return DDPrec;
end

function performSetup(i::Array{Int64}, AI::DomainDecompositionOperatorConstructor,  DDparam::DomainDecompositionParam)
	AII = AI.getOperator(AI.problem_param);
	if AI.getDirichletMass !=identity
		DirichletMass = AI.getDirichletMass(DDparam,AI.problem_param,i)[:];
		AII = AII + sparse(Diagonal(DirichletMass));
	end
	DDPrec = DomainDecompositionPreconditionerParam(AI.problem_param,i,AII,DirichletMass,copySolver(DDparam.Ainv))
	DDPrec.Ainv = setupSolver(AII,DDPrec.Ainv);
	# println("setup for sub domain: ",i," by ",myid());
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



function setupDDSerial(AT::Union{SparseMatrixCSC,DomainDecompositionOperatorConstructor},DDparam::DomainDecompositionParam)
M 			= DDparam.Mesh;
numDomains 	= DDparam.numDomains;
overlap 	= DDparam.overlap;
DDPreconditioners 	= Array{DomainDecompositionPreconditionerParam}(undef,prod(numDomains));
DDparam.GlobalIndices = Array{Array{DDIndType}}(undef,prod(numDomains));
n = M.n;
for ii = 1:prod(numDomains)
	i = cs2loc(ii,numDomains);
	IIp = DDparam.getIndicesOfCell(numDomains,overlap, i,n);
	if isa(AT,SparseMatrixCSC)==true
		AI = sparse(AT[IIp,IIp]');
		DDPreconditioners[ii] = performSetup(i, AI,  DDparam);
	else
		subparams = AT.getSubParams(AT.problem_param, M,i,numDomains,overlap);
		AI = DomainDecompositionOperatorConstructor(subparams,AT.getSubParams,AT.getOperator,AT.getDirichletMass);
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


function solveGSDDSerial(AT::SparseMatrixCSC,b::Array,x::Array,DDparam::DomainDecompositionParam,niter=1,doTranspose::Int64=0)

ncells = DDparam.Mesh.n;
dim = length(ncells);
numDomains = DDparam.numDomains;
overlap = DDparam.overlap;

for k=1:niter
	for ic = 1:prod(numDomains);
		# icloc = cs2loc(ic,numDomains);
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
	for ic = prod(numDomains):-1:1;
		# icloc = cs2loc(ic,numDomains);
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
if niter > 1
	println(k,": ",norm(b - AT'*x)/norm(b))
end
return x,DDparam;
end

function solveSubDomain(r,prec::DomainDecompositionPreconditionerParam,doTranspose::Int64)
	t = zeros(eltype(r),size(r));
	t,prec.Ainv = solveLinearSystem!(prec.A_i,r,t,prec.Ainv,doTranspose);
	return t,prec;
end