
export getHybridKaczmarzPrecond,getHybridKaczmarz
const parRelax_lib  = abspath(joinpath(splitdir(Base.source_path())[1],"../..","deps","builds","parRelax"))


const ArrIdxsType = UInt32;

mutable struct hybridKaczmarz{VAL,IND}
	DDparam 	:: DomainDecompositionParam;
	invDiag 	:: Array
	numCores	:: Int64
	omega_damp	:: Float64
	ArrIdxs 	:: Array{ArrIdxsType,2}
	precond 	:: Function
	numit 		:: Int64
end

function getHybridKaczmarz(VAL::Type,IND::Type,AT::SparseMatrixCSC,Mesh::RegularMesh, numDomains::Array{Int64,1}, 
						getIndicesOfCell::Function,omega_damp::Float64,numCores::Int64,numit::Int64) 
if prod(numDomains) < numCores
	println("*** WARNING: getHybridKaczmarz: numDomains < numCores. ***");
end
DDparam = getDomainDecompositionParam(Mesh,numDomains,zeros(Int64,size(numDomains)),getIndicesOfCell, getJuliaSolver());
invDiag = convert(Array{VAL,1},omega_damp./(vec(sum(conj(AT).*AT,dims=1))));
ArrIdxs = getIndicesOfCellsArray(DDparam);
return hybridKaczmarz{VAL,IND}(DDparam,invDiag,numCores,omega_damp,ArrIdxs,identity,numit);
end

function getHybridKaczmarzPrecond(param::hybridKaczmarz{VAL,IND},AT::SparseMatrixCSC{VAL,IND},nrhs::Int64) where {VAL,IND}
# void applyHybridKaczmarz(spIndType *rowptr , spValType *valA ,spIndType *colA, long long numDomains, long long domainLength ,
						# unsigned int *ArrIdxs, spValType *x, spValType *b, spValType *invD,long long numit, long long numCores){
x = zeros(VAL,size(AT,2),nrhs);
numDomains = prod(param.DDparam.numDomains);
param.precond = (r)->(x[:].=0.0;applyHybridKaczmarz(param::hybridKaczmarz{VAL,IND},AT::SparseMatrixCSC{VAL,IND},r::Array{VAL},x::Array{VAL},numDomains);return x;);
return param.precond; 
end

function applyHybridKaczmarz(param::hybridKaczmarz{Float64,Int64},AT::SparseMatrixCSC{Float64,Int64},r::Array{Float64},x::Array{Float64},numDomains)
	ccall((:applyHybridKaczmarz_FP64_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{Float64},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{Float64},Ptr{Float64},Int64,Int64,Ptr{Float64},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

function applyHybridKaczmarz(param::hybridKaczmarz{Float32,Int64},AT::SparseMatrixCSC{Float32,Int64},r::Array{Float32},x::Array{Float32},numDomains)
	ccall((:applyHybridKaczmarz_FP32_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{Float32},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{Float32},Ptr{Float32},Int64,Int64,Ptr{Float32},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

function applyHybridKaczmarz(param::hybridKaczmarz{ComplexF64,Int64},AT::SparseMatrixCSC{ComplexF64,Int64},r::Array{ComplexF64},x::Array{ComplexF64},numDomains)
	ccall((:applyHybridKaczmarz_CFP64_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{ComplexF64},Ptr{ComplexF64},Int64,Int64,Ptr{ComplexF64},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

function applyHybridKaczmarz(param::hybridKaczmarz{ComplexF32,Int64},AT::SparseMatrixCSC{ComplexF32,Int64},r::Array{ComplexF32},x::Array{ComplexF32},numDomains)
	ccall((:applyHybridKaczmarz_CFP32_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{ComplexF32},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{ComplexF32},Ptr{ComplexF32},Int64,Int64,Ptr{ComplexF32},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

# function applyHybridKaczmarz(param::hybridKaczmarz{VAL,IND},AT::SparseMatrixCSC{VAL,IND},b::Array{VAL},x::Array{VAL},numDomains)
	# ccall((:applyHybridKaczmarz,parRelax_lib),Nothing,(Ptr{Int64},Ptr{spValType},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{spValType},Ptr{spValType},Ptr{spValType},Int64,Int64,),
		# AT.colptr,AT.nzval,AT.rowval,numDomains,size(ArrIdxs,1),param.ArrIdxs,x,r,param.invDiag,param.numit,param.numCores);
# end


