export RelaxVankaFaces,RelaxVankaFacesColor,getVankaFacesPreconditioner,cellColor,cellRBColor

export getHybridCellWisePrecond,getHybridCellWiseParam

export GetElasticityOperatorMixedFormulation

const Vanka_lib     = abspath(joinpath(splitdir(Base.source_path())[1],"../..","deps","builds","Vanka"))


const FULL_VANKA = 1
const KACMARZ_VANKA = 2
const ECON_VANKA = 3

function toSingle(VAL::Type)
if VAL==Float64
	return Float32;
elseif VAL==ComplexF64
	return ComplexF32
else
	return VAL;
end
end


function getVankaVariablesOfCell(i::Array{Int64},n::Array{Int64},nf::Array{Int64},Idxs::Array,includePressure::Bool)
if includePressure
	if length(i)==2	
		@inbounds t1 = i[1] + (i[2]-1)*(n[1]+1);
		@inbounds t2 = nf[1] + i[1] + (i[2]-1)*n[1];
		Idxs[1] = t1;
		Idxs[2] = t1+1;
		Idxs[3] = t2;
		@inbounds Idxs[4] = t2 + n[1];
		@inbounds Idxs[5] = nf[2] + t2;
	else
		@inbounds t1 = loc2cs3D(i,n + [1;0;0]);
		Idxs[1] = t1;
		Idxs[2] = t1+1;
		@inbounds t2 = nf[1] + loc2cs3D(i,n + [0;1;0]);
		Idxs[3] = t2;
		@inbounds Idxs[4] = t2+n[1];
		@inbounds t3 = nf[1] + nf[2] + loc2cs3D(i,n); # it should be for n + [0;0;1] but it is the same for n
		Idxs[5] = t3;
		@inbounds Idxs[6] = t3+n[1]*n[2];
		@inbounds Idxs[7] = nf[3] + t3;
	end
else
	if length(i)==2	
		@inbounds t1 = i[1] + (i[2]-1)*(n[1]+1);
		@inbounds t2 = nf[1] + i[1] + (i[2]-1)*n[1];
		Idxs[1] = t1;
		Idxs[2] = t1+1;
		Idxs[3] = t2;
		@inbounds Idxs[4] = t2 + n[1];
	else
		@inbounds t1 = loc2cs3D(i,n + [1;0;0]);
		Idxs[1] = t1;
		Idxs[2] = t1+1;
		@inbounds t2 = nf[1] + loc2cs3D(i,n + [0;1;0]);
		Idxs[3] = t2;
		@inbounds Idxs[4] = t2+n[1];
		@inbounds t3 = nf[1] + nf[2] + loc2cs3D(i,n); # it should be for n + [0;0;1] but it is the same for n
		Idxs[5] = t3;
		@inbounds Idxs[6] = t3 + n[1]*n[2];
	end
end

# getVankaVariablesOfCell(long long *i,long long *n,long long *nf,long long *Idxs,includePressure, long long dim)
# Idxs2 = copy(Idxs);
# ccall((:getVankaVariablesOfCell,Vanka_lib),Void,(Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Int64, Int64,),i,n,nf,Idxs2,convert(Int64,includePressure),length(n));
# if sum(abs(Idxs - Idxs2)) != 0
	# error("Fix getVankaVariablesOfCell in C");
# end
return Idxs;
end




function cellRBColor(i::Array{Int64})
return mod(sum(i),2)+1;
end


function cellColor(i::Array{Int64})
color = 0;
if length(i)==2
	if mod(i[1],2)==1 
		color =  mod(i[2],2)==1 ? 1 : 2; 
	else 
		color =  mod(i[2],2)==1 ? 3 : 4; 
	end
else
	if mod(i[1],2)==1
		if mod(i[2],2)==1
			color =  mod(i[3],2)==1 ? 1 : 2; 
		else
			color =  mod(i[3],2)==1 ? 3 : 4;
		end	
	else 
		if mod(i[2],2)==1
			color =  mod(i[3],2)==1 ? 5 : 6; 
		else
			color =  mod(i[3],2)==1 ? 7 : 8;
		end
 	end
end

# cellColor(long long *i,long long dim)
# ccolor = ccall((:cellColor,Vanka_lib),Int16,(Ptr{Int64},Int64,),i,length(i));
# if color != ccolor
	# error("Fix cellColor in C");
# end		
return color;
end


# n = 100;
# A = sprandn(n,n,10/n) + 1im*sprandn(n,n,10/n);
# AT = A';

# for k=1:20
	# I = sort(randperm(n)[1:10]);
	# Acc = full(AT[I,I])';
	# AccNew = getDenseBlockFromAT(AT,I);
	# println(vecnorm(Acc-AccNew))
# end

function getDenseBlockFromAT(AT::SparseMatrixCSC{VAL,IND},Idxs::Array,Acc::Array{VAL,2}) where {VAL,IND}
	Acc[:] .= 0.0;
	for t = 1:length(Idxs)
		ii = AT.colptr[Idxs[t]];
		jj = 1;		
		while ii < AT.colptr[Idxs[t]+1] && jj <= length(Idxs)  
			if AT.rowval[ii] ==  Idxs[jj]
				Acc[t,jj] = conj(AT.nzval[ii]);
				ii+=1;
				jj+=1;
			elseif AT.rowval[ii] >  Idxs[jj]
				jj+=1;
			else
				ii+=1;
			end
		end
	end
	return Acc;
end



# function getDenseBlockFromAAT(AT::SparseMatrixCSC,Idxs::Array,Acc::Array)
	# Acc[:] = 0.0;
	# for t = 1:length(Idxs)
		# ii = AT.colptr[Idxs[t]];
		# jj = 1;
		# while ii < AT.colptr[Idxs[t]+1] && jj <= length(Idxs)  
			# if AT.rowval[ii] ==  Idxs[jj]
				# Acc[t,jj] = conj(AT.nzval[ii]);
				# ii+=1;
				# jj+=1;
			# elseif AT.rowval[ii] >  Idxs[jj]
				# jj+=1;
			# else
				# ii+=1;
			# end
		# end
	# end
	# return Acc;
# end


function computeResidualAtIdx(AT::SparseMatrixCSC{VAL,IND},b::Array{VAL},x::Array{VAL},Idxs::Array{IND}) where {VAL,IND}
r = b[Idxs];
for t = 1:length(Idxs)
	for tt = AT.colptr[Idxs[t]]:AT.colptr[Idxs[t]+1]-1
		r[t] = r[t] - conj(AT.nzval[tt])*x[AT.rowval[tt]];
	end
end

#void computeResidualAtIdx(long long *rowptr , double complex *valA ,long long *colA,double complex *b,double complex *x,long long* Idxs, double complex *local_r, int blockSize){
# local_r = r*0.0;
# ccall((:computeResidualAtIdx,lib),Void,(Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Ptr{Complex128},Ptr{Complex128},Ptr{Int64},Ptr{Complex128},Int16,)
								 # ,AT.colptr,AT.nzval,AT.rowval,b,x,Idxs,local_r,convert(Int16,length(Idxs)));
# if norm(local_r - r) > 1e-14
	# error("Fix computeResidualAtIdx in C: ",norm(local_r - r));
# end
return r;
end


function getVankaBlockSize(n::Array{Int64},includePressure::Bool)
blockSize = 0;
nf = 0;
if length(n)==2
	nf = [prod(n + [1; 0]),prod(n + [0; 1])];
	blockSize = includePressure ? 5 : 4;
else
	nf = [prod(n + [1; 0; 0]),prod(n + [0; 1; 0]),prod(n + [0; 0; 1])];
	blockSize = includePressure ? 7 : 6;	
end
return blockSize,nf;
end

function setupVankaFacesPreconditioner(AT::SparseMatrixCSC{VAL,IND},M::RegularMesh,w::Float64,includePressure::Bool,VankaType::Int64 = FULL_VANKA) where {VAL,IND}
n = M.n;
vankaPrecType = toSingle(VAL);
blockSize,nf = getVankaBlockSize(n,includePressure);
numVankaVarPerCell = blockSize*blockSize;
if VankaType == ECON_VANKA
	numVankaVarPerCell = blockSize + 2*(blockSize-1);
end
LocalBlocks = zeros(vankaPrecType,numVankaVarPerCell,prod(M.n));
Acc = zeros(eltype(AT),blockSize,blockSize);
Idx_i = zeros(spIndType,blockSize);
for ii = 1:prod(n)
	i = cs2loc(ii,n);
	Idx_i = getVankaVariablesOfCell(i,n,nf,Idx_i,includePressure);
	## THESE LINES ARE REALLY SLOW
	# Acc1 = AT[Idx_i,Idx_i];
	# Acc1 = full(Acc1');
	
	if VankaType == FULL_VANKA
		Acc = getDenseBlockFromAT(AT,Idx_i,Acc)
		AccInv = convert(Array{vankaPrecType},(w.*inv(Acc))');
	elseif VankaType == KACMARZ_VANKA
		ATi = convert(SparseMatrixCSC{ComplexF64,Int64},AT[:,Idx_i]);
		AccInv = convert(Array{vankaPrecType},(w.*inv(Matrix(ATi'*ATi)))');
	elseif VankaType == ECON_VANKA
		Acc = getDenseBlockFromAT(AT,Idx_i,Acc);
		AccInv = [diag(Acc);Acc[1:end-1,end];Acc[end,1:end-1]];
	else
		error("unknown Vanka Type.")
	end
	LocalBlocks[:,ii] = AccInv[:];
end
return LocalBlocks;
end

function RelaxVankaFacesColor(AT::SparseMatrixCSC{VAL,IND},x::Array{VAL},b::Array{VAL},y::Array{VAL},
								D::Array,numit::Int64,numCores::Int64,M::RegularMesh,
								includePressure::Bool) where {VAL,IND}
# function RelaxVankaFaces(AT::SparseMatrixCSC,M::RegularMesh)
	if toSingle(VAL)!=eltype(D)
		error("check types.");
	end
	n = M.n;
	dim = M.dim;
	blockSize,nf = getVankaBlockSize(n,includePressure);
	parallel = true;
	if parallel==false
		Idxs = zeros(Int64,blockSize);
		# y_t = copy(x);
		for k=1:numit
			for color = 1:(2^dim)
				y[:] = x;
				for i = 1:prod(n)
					i_vec = cs2loc(i,n);
					if cellColor(i_vec)==color
						Idxs = getVankaVariablesOfCell(i_vec,n,nf,Idxs,includePressure);
						r = computeResidualAtIdx(AT,b,x,Idxs);
						x[Idxs] = x[Idxs] + (reshape(D[:,i],blockSize,blockSize)'*r);
						# void updateSolution(float complex *mat, double complex *x, double complex *r, int n,long long* Idxs){
						# ccall((:updateSolution,Vanka_lib),Void,(Ptr{vankaPrecType},Ptr{Complex128},Ptr{Complex128},Int16,Ptr{Int64},)
									# ,D[:,i],y ,r,convert(Int16,blockSize),Idxs);
						# if norm(x[Idxs] - y[Idxs]) > 1e-14
						# error("Fix updateSolution in C: ",norm(x[Idxs] - y[Idxs]));
						# end
					
					end
				end
			end
		end
	else
		y[:] .= 0.0;
		applyVankaFacesColor(AT,x,b,y,D,numit,numCores,n,nf,dim,convert(Int64,includePressure));
	end
	# if norm(x - y_t) > 1e-14
		# error("Fix RelaxVankaFacesColor in C: ",norm(x - y_t));
	# end
	# println("Diff: ",norm(x-y_t));
	return x;
end
# GET_VANKA(ValName,IndName)(spIndType *rowptr , spValType *valA ,spIndType *colA,long long *n,long long *nf,long long dim,
							# spValType *x, spValType *b, spValType *y, vankaPrecType *D,long long numit,long long includePressure,
							# long long lengthVecs, long long numCores){
# function applyVankaFacesColor(AT::SparseMatrixCSC{VAL,IND},x::Array{VAL},b::Array{VAL},y::Array{VAL},
								# D::Array,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Int64)
	# ccall((:RelaxVankaFacesColor_FP64_INT64,Vanka_lib),Nothing,(Ptr{IND},Ptr{VAL},Ptr{IND},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{VAL},Ptr{VAL},Ptr{VAL},Ptr{toSingle(VAL)},Int64, Int64, Int64,Int64,),
			# AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,y,D,numit,convert(Int64,includePressure),length(x),numCores);
# end

function applyVankaFacesColor(AT::SparseMatrixCSC{Float64,Int64},x::Array{Float64},b::Array{Float64},y::Array{Float64},
								D::Array,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Int64)
	ccall((:RelaxVankaFacesColor_FP64_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{Float64},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{toSingle(Float64)},Int64, Int64, Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,y,D,numit,includePressure,length(x),numCores);
end

function applyVankaFacesColor(AT::SparseMatrixCSC{ComplexF64,Int64},x::Array{ComplexF64},b::Array{ComplexF64},y::Array{ComplexF64},
								D::Array,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Int64)
	ccall((:RelaxVankaFacesColor_FP64_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{toSingle(ComplexF64)},Int64, Int64, Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,y,D,numit,convert(Int64,includePressure),length(x),numCores);
end

function applyVankaFacesColor(AT::SparseMatrixCSC{ComplexF32,Int64},x::Array{ComplexF32},b::Array{ComplexF32},y::Array{ComplexF32},
								D::Array,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Int64)
	ccall((:RelaxVankaFacesColor_FP64_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{ComplexF32},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{ComplexF32},Ptr{ComplexF32},Ptr{ComplexF32},Ptr{toSingle(ComplexF32)},Int64, Int64, Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,y,D,numit,convert(Int64,includePressure),length(x),numCores);
end





function getHybridCellWiseParam(AT::SparseMatrixCSC{VAL,Int64},Mesh::RegularMesh, numDomains::Array{Int64,1},
								omega::Float64,numCores::Int64,numit::Int64,mixed::Bool,Kaczmarz::Bool) where {VAL,Int64}
if prod(numDomains) < numCores
	println("WARNING: getHybridKaczmarz: numDomains < numCores.");
end
DDparam = getDomainDecompositionParam(Mesh,numDomains,zeros(Int64,size(numDomains)),getCellCenteredIndicesOfCell);
invDiag = setupVankaFacesPreconditioner(AT,Mesh,omega,mixed,Kaczmarz);
if Kaczmarz
	ArrIdxs = getIndicesOfCellsArray(DDparam);
else
	ArrIdxs = zeros(UInt32,0,0);
end
return hybridKaczmarz{VAL,IND}(DDparam,invDiag,numCores,omega,ArrIdxs,identity,numit);
end



# function getHybridCellWisePrecond(param::hybridKaczmarz{VAL,IND},AT::SparseMatrixCSC{VAL,IND},x::Array{VAL},includePressure::Bool,Kaczmarz::Bool) where {VAL,IND}
# M = param.DDparam.Mesh;
# invDiag = param.invDiag;
# numCores = param.numCores;
# numit = param.numit;
# ArrIdxs = param.ArrIdxs;
# numDomains = param.DDparam.numDomains;
# n = M.n;
# dim = M.dim;
# blockSize = 0;
# nf = 0;
# if dim==2
	# nf = [prod(n + [1; 0]),prod(n + [0; 1])];
	# blockSize = includePressure ? 5 : 4;
# else
	# # Face sizes
	# nf = [prod(n + [1; 0; 0]),prod(n + [0; 1; 0]),prod(n + [0; 0; 1])];
	# blockSize = includePressure ? 7 : 6;
# end
# if Kaczmarz
	# param.precond = (r)->(x[:] = 0.0; ccall((:applyHybridCellWiseKaczmarz,Vanka_lib),Nothing,(Ptr{Int64},Ptr{VAL},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{VAL},Ptr{VAL},Ptr{VAL},Int64, Int64, Int64,Ptr{UInt32},Int64,Int64,),
			# AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,r,invDiag,numit,convert(Int64,includePressure),numCores,ArrIdxs,prod(numDomains),size(ArrIdxs,1)); return x;);
# else ## Vanka
	# param.precond = (r)->(x[:] = 0.0; ccall((:RelaxVankaFacesColor,Vanka_lib),Nothing,(Ptr{Int64},Ptr{VAL},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{VAL},Ptr{VAL},Ptr{VAL},Ptr{VAL},Int64, Int64, Int64,Int64,),
			# AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,r,x,invDiag,numit,convert(Int64,includePressure),length(x),numCores); return x;);
# end
# return param.precond; 
# end










