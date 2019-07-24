
export getHybridKaczmarzPrecond,getHybridKaczmarz
const parRelax_lib  = abspath(joinpath(splitdir(Base.source_path())[1],"..","deps","builds","parRelax"))

mutable struct hybridKaczmarz
	DDparam :: DomainDecompositionParam;
	invDiag :: Array
	numCores:: Int64
	omega	:: Float64
	ArrIdxs :: Array{UInt32,2}
	precond :: Function
	numit 	:: Int64
end

function getHybridKaczmarz(AT::SparseMatrixCSC,Mesh::RegularMesh, numDomains::Array{Int64,1}, 
						getIndicesOfCell::Function,omega::Float64,numCores::Int64,numit::Int64)
if prod(numDomains) < numCores
	warn("WARNING: getHybridKaczmarz: numDomains < numCores.");
end
DDparam = getDomainDecompositionParam(Mesh,numDomains,zeros(Int64,size(numDomains)),getIndicesOfCell);
invDiag = convert(Vector{eltype(AT)},omega./(vec(sum(conj(AT).*AT,1))));
ArrIdxs = getIndicesOfCellsArray(DDparam);
precond = (r)->copy(r);
return hybridKaczmarz(DDparam,invDiag,numCores,omega,ArrIdxs,identity,numit);
end

function getHybridKaczmarzPrecond(param::hybridKaczmarz,AT::SparseMatrixCSC{spValType,Int64},b::Array{spValType})
# void applyHybridKaczmarz(spIndType *rowptr , spValType *valA ,spIndType *colA, long long numDomains, long long domainLength ,
						# unsigned int *ArrIdxs, spValType *x, spValType *b, spValType *invD,long long numit, long long numCores){
x = copy(b);
invDiag = param.invDiag;
numCores = param.numCores;
numit = param.numit;
ArrIdxs = param.ArrIdxs;
numDomains = param.DDparam.numDomains;
param.precond = (r)->(x[:] = 0.0; ccall((:applyHybridKaczmarz,parRelax_lib),Nothing,(Ptr{Int64},Ptr{spValType},Ptr{Int64},Int64,Int64,Ptr{UInt32},Ptr{spValType},Ptr{spValType},Ptr{spValType},Int64,Int64,),
					AT.colptr,AT.nzval,AT.rowval,prod(numDomains),size(ArrIdxs,1),ArrIdxs,x,r,invDiag,numit,numCores); return x;);
return param.precond; 
end