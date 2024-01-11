
export getHybridKaczmarzPrecond,getHybridKaczmarz, getHybridKaczmarzParam
const parRelax_lib  = abspath(joinpath(splitdir(Base.source_path())[1],"../..","deps","builds","parRelax"))


const ArrIdxsType = UInt32;

mutable struct hybridKaczmarz{VAL,IND}
	numDomains 	:: Array{Int64,1}
	invDiag 	:: Array
	numCores	:: Int64
	omega_damp	:: Float64
	ArrIdxs 	:: Array{ArrIdxsType,2}
	precond 	:: Function
	numit 		:: Int64
	getIndicesOfCell::Function
end

# import Base.copy
# function copy(p::hybridKaczmarzParams)
	# return hybridKaczmarzParams(p.numDomains,p.numCores,p.omega_damp,p.numit);
# end

function getHybridKaczmarz(VAL::Type,IND::Type,numDomains::Array{Int64,1}, 
						getIndicesOfCell::Function,omega_damp::Float64,numCores::Int64,numit::Int64) 
if prod(numDomains) < numCores
	println("*** WARNING: getHybridKaczmarz: numDomains < numCores. ***");
end
return hybridKaczmarz{VAL,IND}(numDomains,[],numCores,omega_damp,zeros(ArrIdxsType,1,1),identity,numit,getIndicesOfCell);
end

function setupHybridKaczmarz(param::hybridKaczmarz{VAL,IND},AT::SparseMatrixCSC,Mesh::RegularMesh) where {VAL,IND}
param.invDiag = convert(Array{VAL,1},param.omega_damp./(vec(sum(conj(AT).*AT,dims=1))));
param.ArrIdxs = getIndicesOfCellsArray(Mesh, zeros(Int64,size(param.numDomains)),param.numDomains,param.getIndicesOfCell);
return param;
end


function getHybridKaczmarz(VAL::Type,IND::Type,AT::SparseMatrixCSC,Mesh::RegularMesh, numDomains::Array{Int64,1}, 
						getIndicesOfCell::Function,omega_damp::Float64,numCores::Int64,numit::Int64) 
if prod(numDomains) < numCores
	println("*** WARNING: getHybridKaczmarz: numDomains < numCores. ***");
end
invDiag = convert(Array{VAL,1},omega_damp./(vec(sum(conj(AT).*AT,dims=1))));
ArrIdxs = getIndicesOfCellsArray(Mesh, zeros(Int64,size(numDomains)),numDomains,getIndicesOfCell);
return hybridKaczmarz{VAL,IND}(numDomains,invDiag,numCores,omega_damp,ArrIdxs,identity,numit,getIndicesOfCell);
end

function getHybridKaczmarzPrecond(param::hybridKaczmarz{VAL,IND},AT::SparseMatrixCSC{VAL,IND},nrhs::Int64) where {VAL,IND}
# void applyHybridKaczmarz(spIndType *rowptr , spValType *valA ,spIndType *colA, long long numDomains, long long domainLength ,
						# unsigned int *ArrIdxs, spValType *x, spValType *b, spValType *invD,long long numit, long long numCores){
x = zeros(VAL,size(AT,2),nrhs);
if nrhs == 1
	x = vec(x);
end
numDomains = prod(param.numDomains);
param.precond = (r)->(x[:].=0.0;applyHybridKaczmarz(param::hybridKaczmarz{VAL,IND},AT,r,x,numDomains);return x;);
return param.precond; 
end

function applyHybridKaczmarz(param::hybridKaczmarz{Float64,Int64},AT::SparseMatrixCSC{Float64,Int64},r::Array{Float64},x::Array{Float64},numDomains::Int64)
	ccall((:applyHybridKaczmarz_FP64_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{Float64},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{Float64},Ptr{Float64},Int64,Int64,Ptr{Float64},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

function applyHybridKaczmarz(param::hybridKaczmarz{Float32,Int64},AT::SparseMatrixCSC{Float32,Int64},r::Array{Float32},x::Array{Float32},numDomains::Int64)
	ccall((:applyHybridKaczmarz_FP32_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{Float32},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{Float32},Ptr{Float32},Int64,Int64,Ptr{Float32},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

function applyHybridKaczmarz(param::hybridKaczmarz{ComplexF64,Int64},AT::SparseMatrixCSC{ComplexF64,Int64},r::Array{ComplexF64},x::Array{ComplexF64},numDomains::Int64)
	ccall((:applyHybridKaczmarz_CFP64_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{ComplexF64},Ptr{ComplexF64},Int64,Int64,Ptr{ComplexF64},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

function applyHybridKaczmarz(param::hybridKaczmarz{ComplexF32,Int64},AT::SparseMatrixCSC{ComplexF32,Int64},r::Array{ComplexF32},x::Array{ComplexF32},numDomains::Int64)
	ccall((:applyHybridKaczmarz_CFP32_INT64,parRelax_lib),Nothing,(Ptr{Int64},Ptr{ComplexF32},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{ComplexF32},Ptr{ComplexF32},Int64,Int64,Ptr{ComplexF32},Int64,Int64,),
		AT.colptr,AT.nzval,AT.rowval,numDomains,size(param.ArrIdxs,1),param.ArrIdxs,x,r,size(r,2),size(r,1),param.invDiag,param.numit,param.numCores);
end

# function applyHybridKaczmarz(param::hybridKaczmarz{VAL,IND},AT::SparseMatrixCSC{VAL,IND},b::Array{VAL},x::Array{VAL},numDomains)
	# ccall((:applyHybridKaczmarz,parRelax_lib),Nothing,(Ptr{Int64},Ptr{spValType},Ptr{Int64},Int64,Int64,Ptr{ArrIdxsType},Ptr{spValType},Ptr{spValType},Ptr{spValType},Int64,Int64,),
		# AT.colptr,AT.nzval,AT.rowval,numDomains,size(ArrIdxs,1),param.ArrIdxs,x,r,param.invDiag,param.numit,param.numCores);
# end


