export setupDDParallel
export getWorkerForSubDomainMultiColor,solveDDParallel


function setupDDParallel(AT::Union{SparseMatrixCSC,DomainDecompositionOperatorConstructor{VAL,IND}},DDparam::DomainDecompositionParam{VAL,IND},workerList::Array{Int64}) where {VAL,IND}
if workerList==[]
	return setupDDSerial(AT,DDparam);
else
	ActualWorkers = intersect(workerList,workers());
	if length(ActualWorkers)<length(workerList)
		warn("setupDDParallel: workerList included indices of non-existing workers.")
	end
end
numWorkers = length(ActualWorkers);

M 			= DDparam.Mesh;
numDomains 	= DDparam.numDomains;
overlap 	= DDparam.overlap;
n = M.n;



DDPreconditioners 	= Array{RemoteChannel}(undef,prod(numDomains));
DDparam.GlobalIndices = Array{Array{DDIndType}}(undef,prod(numDomains));

# ix = 1; nextidx() = (idx=ix; ix+=1; idx)

@sync begin
	for p = ActualWorkers
		@async begin
			# while true
				# idx = nextidx()
				# if idx > numWorkers
					# break
				# end
				for ii=1:length(DDPreconditioners)
					i = cs2loc(ii,numDomains);
					worker_index = getWorkerForSubDomainMultiColor(i,numDomains,numWorkers);
					
					# println("For cell ",i," we need worker ",ActualWorkers[worker_index],", and we have ",p)
					if ActualWorkers[worker_index] == p
						IIp = DDparam.getIndicesOfCell(numDomains,overlap, i,n);
						subMesh = getSubMeshOfCell(numDomains,overlap,i,M);
						if isa(AT,SparseMatrixCSC)==true
							AI = sparse(AT[IIp,IIp]');
							DDPreconditioners[ii] = initRemoteChannel(performSetup,p,i, AI,  DDparam,subMesh);
						else
							subparams = AT.getSubParams(AT.problem_param, M,i,numDomains,overlap);
							AI = DomainDecompositionOperatorConstructor(subparams,AT.getSubParams,AT.getOperator,AT.getDirichletMass);
							DDPreconditioners[ii] = initRemoteChannel(performSetup,p,i, AI,  DDparam,subMesh);
						end
						IIp = convert(Array{DDIndType},IIp);
						DDparam.GlobalIndices[ii] = IIp;
						wait(DDPreconditioners[ii]);
					end
				end
			# end
		end
	end
end
DDparam.PrecParams = DDPreconditioners;
return DDparam;
end


function solveDDParallel(AT::SparseMatrixCSC,b::Array,x::Array,DDparam::DomainDecompositionParam{VAL,IND},workerList,niter=1,doTranspose::Int64=0) where {VAL,IND}
if workerList==[]
	return solveDD(AT,b,x,DDparam,niter,doTranspose);
else
	ActualWorkers = intersect(workerList,workers());
	if length(ActualWorkers)<length(workerList)
		error("setupDDParallel: workerList included indices of non-existing workers:",workerList,",",ActualWorkers)
	end
end
numWorkers = length(ActualWorkers);


numDomains 	= DDparam.numDomains;
overlap 	= DDparam.overlap;
dim 		= length(DDparam.numDomains);

for k=1:niter
	for color = 1:2^dim
	@sync begin
			# ix = 1; nextidx() = (idx=ix; ix+=1; idx)
			for p = ActualWorkers
				@async begin
					# while true
						# idx = nextidx()
						# if idx > numWorkers
							# break
						# end
						for ii=1:prod(numDomains)
							i = cs2loc(ii,numDomains);
							if ActualWorkers[getWorkerForSubDomainMultiColor(i,numDomains,numWorkers)] == p && cellColor(i)==color
								IIp = DDparam.GlobalIndices[ii];
								r = computeResidualAtIdx(AT,b,x,IIp);
								if DDparam.PrecParams[ii].where != p
									error("Sending to wrong worker!!!");
								end
								# println("before sending worker ",p," to solve subdomain",i,", for color ",cellColor(i))
								(t,DDparam.PrecParams[ii]) = remotecall_fetch(solveSubDomain, p, r, DDparam.PrecParams[ii],doTranspose);
								x[IIp] = x[IIp] + t;
								# println("after sending worker ",p," to solve subdomain",i,", for color ",cellColor(i))
							end
						end
					# end
				end
			end
		end
	end
	if niter > 1
		println(k,": ",norm(b - AT'*x)/norm(b))
	end
end
return x,DDparam;
end


function solveSubDomain(r,precRef::RemoteChannel,doTranspose::Int64)
	prec = take!(precRef);
	t = zeros(eltype(r),size(r));
	(t,prec.Ainv) = solveLinearSystem!(prec.A_i,r,t,prec.Ainv,doTranspose);
	precRef = put!(precRef,prec)
	return (t,precRef)
end



function getWorkerForSubDomainMultiColor(iiloc::Array{Int64},NumCells,NumWorkers)
	NumCellsWorkers = div.(NumCells.+1,2);
	iiploc = div.(iiloc.+1,2);
	iip = loc2cs(iiploc,NumCellsWorkers);
	p = mod(iip-1,NumWorkers)+1;
	return p;
end
