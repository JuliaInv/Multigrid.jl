


# export setupDDParallel
# function setupDDParallel(AT::Union{SparseMatrixCSC,DomainDecompositionOperatorConstructor},M::RegularMesh,getIndicesOfCell::Function,NumCells::Array{Int64},overlap::Array{Int64},workerList::Array{Int64},dropTol = 1e-3)
# if workerList==[]
	# retrun setupDD(AT,M,getIndicesOfCell,NumCells,overlap,dropTol);
# else
	# ActualWorkers = intersect(workerList,workers());
	# if length(ActualWorkers)<length(workerList)
		# warn("setupDDParallel: workerList included indices of non-existing workers.")
	# end
# end
# numWorkers = length(ActualWorkers);
# DDFactors  = Array(Future,prod(NumCells));
# @sync begin
	# for p=ActualWorkers
		# @async begin
			# while true
				# idx = nextidx()
				# if idx > numWorkers
					# break
				# end
				# for ii=1:length(DDFactors)
					# if getWorker(ii,NumCells,numWorkers) == p
						# i = cs2loc(ii,NumCells);
						# if isa(AT,SparseMatrixCSC)==true
							# IIp = getIndicesOfCell(NumCells,overlap, i,n);
							# AI = AT[IIp,IIp]';
							# DDFactors[ii] = remotecall_wait(performSetup,p, AI)
						# else
							# subparams = AT.getSubParams(AT.param, M,i,NumCells,overlap);
							# DDFactors[ii] = remotecall_wait(performSetup,p, AT.getOperator,subparams);
						# end
					# end
				# end
			# end
		# end
	# end
# end
# end

# export getWorker
# function getWorker(ii,NumCells,NumWorkers)
	# NumCellsWorkers = div(NumCells+1,2);
	# iiloc = cs2loc(ii,NumCells);
	# iiploc = div(iiloc+1,2);
	# iip = loc2cs(iiploc,NumCellsWorkers);
	# p = mod(iip,NumWorkers);
	# return p;
# end

# function solveDDParallel(AT::SparseMatrixCSC,n::Array{Int64},b::ArrayTypes,x::ArrayTypes,getIndicesOfCell::Function,NumCells::Array{Int64,1},overlap::Array{Int64,1},DDfactors::Array{Any,1},niter = 1,doTranspose::Int64=0)

# end