export RelaxVankaFacesColor,RelaxHybridVanka,getVankaFacesPreconditioner,cellColor,cellRBColor,getHybridVankaFaces

export getHybridCellWisePrecond,getHybridCellWiseParam

export GetElasticityOperatorMixedFormulation

const Vanka_lib     = abspath(joinpath(splitdir(Base.source_path())[1],"../..","deps","builds","Vanka"))


const parallel = false;

export KACMARZ_VANKA
const FULL_VANKA_RB = 1
const KACMARZ_VANKA = 2
const ECON_VANKA_RB = 3
const FULL_VANKA_LEX = 4
const FULL_VANKA_ADD = 5


function getVankaRelaxType(s::String)
	if s == "VankaFaces"
		return (true,FULL_VANKA_RB);
	elseif s == "EconVankaFaces"
		return (true,ECON_VANKA_RB);
	elseif s == "VankaFacesLex"
		return (true,FULL_VANKA_LEX);
	elseif s == "VankaFacesAdd"
		return (true,FULL_VANKA_ADD);
	else
		return (false,0)
	end
end

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



# function getDenseBlockFromAAT(AT::SparseMatrixCSC{VAL,IND},Idxs::Array,Acc::Array{VAL,2}) where {VAL,IND}
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


# [ D    y ][a0] = [b0] 
# [ x'   z ][a1] = [b1]
# D*a0  + y*a1 = b0
# x'*a0 + z*a1 = b1
# 1) a1 = (b1 - x'*D\b0) / (z-x'*D\y)
# 2) a0 = D\(b0 - y*a1) = D\b0 - a1*D\y

# with w:
# 1) a1 = w*((b1 - x'*D\b0) / (z-x'*D\y))
# 2) a0 = w*(D\(b0 - y*a1)) = w*(D\b0 - a1*D\y)
# which is:
# 1) a1 = ((b1 - x'*D\b0) / ((z-x'*D\y)/w))
# 2) a0 = w*(D\(b0 - (y/w)*a1)) = w*(D\b0 - a1*w*D\y)

# We can save: gamma = w/(z-x'*D\y); beta = D\x; alpha = y/w;  
# a1 = (b1 - beta'*b0)*gamma
# a0 = w*invD.*(b0 - a1*alpha)




# export testEconVanka
# function testEconVanka()
# n=5; 
# D = diagm(rand(n).+ 1e-1) ;
# x = 0.1*rand(n);
# y = 0.1*rand(n);
# z = rand() .+ 1e-1;
# A = [ D x ; y' z]
# w = 0.85;

# beta = rand(n+1);
# alpha = zeros(n+1);

# gamma_test = alpha + w*(A\beta)

# D = convert(Array{Float32},extractEconVankaComponents(A,w));

# gamma = zeros(n+1);
# gamma[end] = (beta[end] - dot(D[1:n],beta[1:n]))*D[n+1];
# gamma[1:n] = (beta[1:n] .- D[2*n+2:end]*gamma[end]).*D[(n+2):(2*n+1)]
# gamma += alpha; 

# display(gamma-gamma_test);println();

# gammaC = copy(alpha);
# Idxs = collect(1:n+1); 
# ccall((:updateSolutionEconomic_FP64,Vanka_lib),Nothing,(Ptr{Float32},Ptr{Float64},Ptr{Float64},Int16,Ptr{Int64},)
									# ,D,gammaC ,beta,convert(Int16,n+1),Idxs);
# display(gammaC-gamma);println();

# # gamma[end] = (beta[end] - y'*(D\beta[1:n]))/(z-y'*(D\x));
# # gamma[1:n] = D\(beta[1:n] - x*alpha[end])
# end





function extractEconVankaComponents(Acc::Array{VAL,2},w) where {VAL}
	d = diag(Acc); z = d[end]; dinv = 1.0./d[1:end-1];
	alpha = Acc[1:end-1,end]; 
	beta  = dinv.*Acc[end,1:end-1];
	z 	  = (z - dot(beta,alpha));
	AccInv = [beta;w/z;w*dinv;alpha/w];
	return AccInv; 
end


function setupVankaFacesPreconditioner(AT::SparseMatrixCSC{VAL,IND},M::RegularMesh,w::Union{Float64,Tuple{Float64,Float64}},includePressure::Bool,VankaType::Int64 = FULL_VANKA_RB) where {VAL,IND}
n = M.n;
vankaPrecType = toSingle(VAL);
blockSize,nf = getVankaBlockSize(n,includePressure);
numVankaVarPerCell = blockSize*blockSize;
if VankaType == ECON_VANKA_RB && parallel == true
	numVankaVarPerCell = 3*(blockSize-1)+1;
end
LocalBlocks = zeros(vankaPrecType,numVankaVarPerCell,prod(M.n));
Acc = zeros(eltype(AT),blockSize,blockSize);
Idx_i = zeros(spIndType,blockSize);
W = ones(blockSize);
if length(w)==2
	W = W*w[1];
	if includePressure
		W[end] = w[2];
	end
else
	W = W*w;
end

if VankaType == KACMARZ_VANKA 
	AAT = conj.(AT'*AT);
else
	AAT = 0.0;
end
for ii = 1:prod(n)
	i = cs2loc(ii,n);
	Idx_i = getVankaVariablesOfCell(i,n,nf,Idx_i,includePressure);
	## THESE LINES ARE REALLY SLOW
	# Acc1 = AT[Idx_i,Idx_i];
	# Acc1 = full(Acc1');
	
	if VankaType == FULL_VANKA_RB || VankaType == FULL_VANKA_LEX
		Acc = getDenseBlockFromAT(AT,Idx_i,Acc)
		if length(w)==1
			# if full
			# AccInv = convert(Array{vankaPrecType},(w.*inv(Acc))');
			# if econ
			Acc[1:end-1,1:end-1] = diagm(diag(Acc[1:end-1,1:end-1]));
			AccInv = convert(Array{vankaPrecType},(w.*inv(Acc))');
		else ## length(w)==2 || VankaType == FULL_VANKA_ADD
			# if full
			AccInv = convert(Array{vankaPrecType},(W.*inv(Acc))');
		end
	elseif VankaType == FULL_VANKA_ADD
		Acc = getDenseBlockFromAT(AT,Idx_i,Acc)
		t = ones(blockSize)*0.5;
		if i[1]==1    t[1] = 1.0;	end
		if i[1]==n[1] t[2] = 1.0;	end
		if i[2]==1    t[3] = 1.0;	end
		if i[2]==n[2] t[4] = 1.0;	end
		if length(i)==3
			if i[3]==1    t[5] = 1.0;	end
			if i[3]==n[3] t[6] = 1.0;	end
		end
		if includePressure
			t[end] = 1.0;
		end
		AccInv = convert(Array{vankaPrecType},((t.*W).*inv(Acc))');
	elseif VankaType == KACMARZ_VANKA
		# ATi = convert(SparseMatrixCSC{ComplexF64,Int64},AT[:,Idx_i]);
		# AccInv = convert(Array{vankaPrecType},(w.*inv(Matrix(ATi'*ATi))));
		Acc = conj.(getDenseBlockFromAT(AAT,Idx_i,Acc));
		AccInv = convert(Array{vankaPrecType},(w.*inv(Acc)'));
	elseif VankaType == ECON_VANKA_RB
		Acc = getDenseBlockFromAT(AT,Idx_i,Acc);
		Acc[1:end-1,1:end-1] = diagm(diag(Acc[1:end-1,1:end-1])./w)
		AccInv = convert(Array{vankaPrecType},(inv(Acc))');
		# AccInv = extractEconVankaComponents(Acc,w);
	else
		error("unknown Vanka Type.")
	end
	LocalBlocks[:,ii] = AccInv[:];
end
return LocalBlocks;
end

function RelaxVankaFacesColor(AT::SparseMatrixCSC{VAL,IND},x::Array{VAL},b::Array{VAL},
								D::Array,numit::Int64,numCores::Int64,M::RegularMesh,
								includePressure::Bool,VankaType::Int64 = FULL_VANKA_RB) where {VAL,IND}
	if toSingle(VAL)!=eltype(D)
		error("check types.");
	end
	n = M.n;
	dim = M.dim;
	
	blockSize,nf = getVankaBlockSize(n,includePressure);
	
	if parallel==false
		Idxs = zeros(Int64,blockSize);
		if VankaType==FULL_VANKA_LEX
			for k=1:numit
				## this is A FULL_VANKA version. 
				for i = 1:prod(n)
					i_vec = cs2loc(i,n);
					Idxs = getVankaVariablesOfCell(i_vec,n,nf,Idxs,includePressure);
					r = computeResidualAtIdx(AT,b,x,Idxs);
					x[Idxs] = x[Idxs] + (reshape(D[:,i],blockSize,blockSize)'*r);
				end
			end
		elseif VankaType==FULL_VANKA_ADD
			y = copy(x);
			for k=1:numit
				## this is A FULL_VANKA version. 
				for i = 1:prod(n)
					i_vec = cs2loc(i,n);
					Idxs = getVankaVariablesOfCell(i_vec,n,nf,Idxs,includePressure);
					r = computeResidualAtIdx(AT,b,y,Idxs);
					x[Idxs] = x[Idxs] + (reshape(D[:,i],blockSize,blockSize)'*r);
				end
			end
		elseif VankaType==FULL_VANKA_RB || VankaType==ECON_VANKA_RB
			y = copy(x);
			for k=1:numit
				## this is A FULL_VANKA version.
				for color = 1:(2^dim)
					y[:] = x;
					for i = 1:prod(n)
						i_vec = cs2loc(i,n);
						if cellColor(i_vec)==color
							Idxs = getVankaVariablesOfCell(i_vec,n,nf,Idxs,includePressure);
							r = computeResidualAtIdx(AT,b,y,Idxs);
							x[Idxs] = x[Idxs] + (reshape(D[:,i],blockSize,blockSize)'*r);
							# void updateSolution(float complex *mat, double complex *x, double complex *r, int n,long long* Idxs){
							# ccall((:updateSolution,Vanka_lib),Void,(Ptr{vankaPrecType},Ptr{Complex128},Ptr{Complex128},Int16,Ptr{Int64},)
									# ,D[:,i],y ,r,convert(Int16,blockSize),Idxs);
						end
					end
				end
			end
		end
	else
		applyVankaFacesColor(AT,x,b,D,numit,numCores,n,nf,dim,convert(Int64,includePressure),VankaType);
	end
	# if norm(x - y_t) > 1e-14
		# error("Fix RelaxVankaFacesColor in C: ",norm(x - y_t));
	# end
	# println("Diff: ",norm(x-y_t));
	return x;
end

function applyVankaFacesColor(AT::SparseMatrixCSC{Float64,Int64},x::Array{Float64},b::Array{Float64},
								D::Array,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Int64,VankaType::Int64)
	ccall((:RelaxVankaFacesColor_FP64_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{Float64},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{Float64},Ptr{Float64},Ptr{toSingle(Float64)},Int64, Int64, Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,D,numit,includePressure,VankaType,numCores);
end

function applyVankaFacesColor(AT::SparseMatrixCSC{ComplexF64,Int64},x::Array{ComplexF64},b::Array{ComplexF64},
								D::Array,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Int64,VankaType::Int64)
	ccall((:RelaxVankaFacesColor_CFP64_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{ComplexF64},Ptr{ComplexF64},Ptr{toSingle(ComplexF64)},Int64, Int64, Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,D,numit,convert(Int64,includePressure),VankaType,numCores);
end

function applyVankaFacesColor(AT::SparseMatrixCSC{ComplexF32,Int64},x::Array{ComplexF32},b::Array{ComplexF32},
								D::Array,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Int64,VankaType::Int64)
	ccall((:RelaxVankaFacesColor_CFP32_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{ComplexF32},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{ComplexF32},Ptr{ComplexF32},Ptr{toSingle(ComplexF32)},Int64, Int64, Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,D,numit,convert(Int64,includePressure),VankaType,numCores);
end


function getHybridVankaFaces(AT::SparseMatrixCSC{VAL,IND},Mesh::RegularMesh, numDomains::Array{Int64,1},
								omega::Float64,numCores::Int64,numit::Int64,mixed::Bool,VankaType::Int64 = KACMARZ_VANKA) where {VAL,IND}
if prod(numDomains) < numCores
	println("WARNING: getHybridKaczmarz: numDomains < numCores.");
end
invDiag = setupVankaFacesPreconditioner(AT,Mesh,omega,mixed,VankaType);
ArrIdxs = getIndicesOfCellsArray(Mesh, zeros(Int64,size(numDomains)),numDomains,getCellCenteredIndicesOfCell);
return hybridKaczmarz{VAL,IND}(numDomains,invDiag,numCores,omega,ArrIdxs,identity,numit);
end



function RelaxHybridVanka(param::hybridKaczmarz{VAL,IND}, AT::SparseMatrixCSC{VAL,IND},x::Array{VAL},b::Array{VAL},
								numit::Int64,numCores::Int64,M::RegularMesh,
								includePressure::Bool,VankaType::Int64 = FULL_VANKA_RB) where {VAL,IND}	
if toSingle(VAL)!=eltype(param.invDiag)
	error("check types.");
end
n = M.n;
dim = M.dim;
blockSize,nf = getVankaBlockSize(n,includePressure);

numCores = param.numCores;
numit = param.numit;
numDomains = param.numDomains;

applyHybridVankaFaces(param, AT,x,b,numit,numCores,n,nf,dim,includePressure,VankaType);
return x;
end


function applyHybridVankaFaces(param::hybridKaczmarz{Float64,Int64}, AT::SparseMatrixCSC{Float64,Int64},x::Array{Float64},b::Array{Float64}
								,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Bool,VankaType::Int64)
	ccall((:applyHybridCellWiseKaczmarz_FP64_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{Float64},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{Float64},Ptr{Float64},Ptr{Float64},Int64, Int64, Int64,Ptr{UInt32},Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,param.invDiag,numit,convert(Int64,includePressure),numCores,param.ArrIdxs,prod(param.numDomains),size(param.ArrIdxs,1));
end

function applyHybridVankaFaces(param::hybridKaczmarz{ComplexF64,Int64}, AT::SparseMatrixCSC{ComplexF64,Int64},x::Array{ComplexF64},b::Array{ComplexF64}
								 ,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Bool,VankaType::Int64)
	 ccall((:applyHybridCellWiseKaczmarz_CFP64_INT64,Vanka_lib),Nothing,(Ptr{Int64},Ptr{ComplexF64},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Int64, Int64, Int64,Ptr{UInt32},Int64,Int64,),
			 AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,param.invDiag,numit,convert(Int64,includePressure),numCores,param.ArrIdxs,prod(param.numDomains),size(param.ArrIdxs,1));
end



# function applyHybridVankaFaces(param::hybridKaczmarz{VAL,IND}, AT::SparseMatrixCSC{VAL,IND},x::Array{VAL},b::Array{VAL}
								# ,numit::Int64,numCores::Int64,n::Array{Int64},nf::Array{Int64},dim::Int64,includePressure::Bool,VankaType::Int64)
	# ccall((:applyHybridCellWiseKaczmarz,Vanka_lib),Nothing,(Ptr{IND},Ptr{VAL},Ptr{IND},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{VAL},Ptr{VAL},Ptr{VAL},Int64, Int64, Int64,Ptr{UInt32},Int64,Int64,),
			# AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,r,param.invDiag,numit,convert(Int64,includePressure),numCores,param.ArrIdxs,prod(param.numDomains),size(param.ArrIdxs,1));
# end




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





