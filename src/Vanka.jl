export RelaxVankaFaces,RelaxVankaFacesColor,getVankaFacesPreconditioner,cellColor,cellRBColor


# const libdir = "C:/Users/YAEL/Dropbox/Eran/JuliaCode/Active/Multigrid.jl/src/";

# const libdir = "C:/Users/etrieste/Dropbox/Eran/JuliaCode/Active/Multigrid.jl/src/";
# const lib = string(libdir,"Vanka.dll");

function loc2cs3D(loc::Array{Int64,1},n::Array{Int64,1})
@inbounds cs = loc[1] + (loc[2]-1)*n[1] + (loc[3]-1)*n[1]*n[2];
return cs;
end

export loc2cs
function loc2cs(loc::Array{Int64,1},n::Array{Int64,1})
if length(n)==2
	@inbounds cs = loc[1] + (loc[2]-1)*n[1];
else
	@inbounds cs = loc[1] + (loc[2]-1)*n[1] + (loc[3]-1)*n[1]*n[2];
end
return cs;
end

function getVankaVariablesOfCell(i::Array{Int64},n::Array{Int64},nf::Array{Int64},Idxs::Array{Int64},includePressure::Bool)
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
# ccall((:getVankaVariablesOfCell,lib),Void,(Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Int64},Int64, Int64,),i,n,nf,Idxs2,convert(Int64,includePressure),length(n));
# if sum(abs(Idxs - Idxs2)) != 0
	# error("Fix getVankaVariablesOfCell in C");
# end
return Idxs;
end


export cs2loc
function cs2loc(cs_loc::Int64,n::Array{Int64,1})
	if length(n)==3
		@inbounds loc1 = mod(cs_loc-1,n[1])+1;
		@inbounds loc2 = div(mod(cs_loc-1,n[1]*n[2]),n[1]) + 1;
		@inbounds loc3 = div(cs_loc-1,n[1]*n[2])+1;
		return [loc1;loc2;loc3];
	else
		@inbounds loc1 = mod(cs_loc-1,n[1]) + 1;
		@inbounds loc2 = div(cs_loc-1,n[1])+1;
		return [loc1;loc2];
	end
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
# ccolor = ccall((:cellColor,lib),Int16,(Ptr{Int64},Int64,),i,length(i));
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

function getDenseBlockFromAT(AT::SparseMatrixCSC,Idxs::Array{Int64},Acc::Array)
	# Acc = zeros(eltype(AT),length(Idxs),length(Idxs));
	Acc[:] = 0.0;
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


const VankaBlockType = Complex64

function getVankaFacesPreconditioner(AT::SparseMatrixCSC,M::RegularMesh,w::Float64,includePressure::Bool)
nf = 0;
n = M.n;
blockSize = 0;
if length(M.n)==2
	nf = [prod(n + [1; 0]),prod(n + [0; 1])];
	blockSize = includePressure ? 5 : 4;
else
	nf = [prod(n + [1; 0; 0]),prod(n + [0; 1; 0]),prod(n + [0; 0; 1])];
	blockSize = includePressure ? 7 : 6;	
end

LocalBlocks = zeros(VankaBlockType,blockSize*blockSize,prod(M.n));

Acc = zeros(eltype(AT),blockSize,blockSize);
Idx_i = zeros(Int64,blockSize);
for ii = 1:prod(M.n)
	i = cs2loc(ii,M.n);
	Idx_i = getVankaVariablesOfCell(i,n,nf,Idx_i,includePressure);
	## THESE LINES ARE REALLY SLOW
	# Acc1 = AT[Idx_i,Idx_i];
	# Acc1 = full(Acc1');
	Acc = getDenseBlockFromAT(AT,Idx_i,Acc)
	AccInv = convert(Array{VankaBlockType},(w.*inv(Acc))');
	LocalBlocks[:,ii] = AccInv[:];
end

return LocalBlocks;
end

# function RelaxVankaFacesColorParallel(AT::SparseMatrixCSC,r::ArrayTypes,x::ArrayTypes,b::ArrayTypes,D::Array{VankaBlockType,2},numit::Int64,numCores::Int64,M::RegularMesh,includePressure::Bool,numCores::Int64)
	# # gcc -O3 -fopenmp -shared -fpic -DBUILD_DLL Vanka.c -o Vanka.dll
	# # const spmatveclib  = abspath(joinpath(splitdir(Base.source_path())[1],"..","deps","builds","ParSpMatVec"))
	
# end


function RelaxVankaFacesColor(AT::SparseMatrixCSC,x::ArrayTypes,b::ArrayTypes,y::ArrayTypes,D::Array{VankaBlockType,2},numit::Int64,numCores::Int64,M::RegularMesh,includePressure::Bool)
# function RelaxVankaFaces(AT::SparseMatrixCSC,M::RegularMesh)
	n = M.n;
	dim = M.dim;
	blockSize = 0;
	nf = 0;
	if dim==2
		nf = [prod(n + [1; 0]),prod(n + [0; 1])];
		blockSize = includePressure ? 5 : 4;
	else
		# Face sizes
		nf = [prod(n + [1; 0; 0]),prod(n + [0; 1; 0]),prod(n + [0; 0; 1])];
		blockSize = includePressure ? 7 : 6;
	end
	Idxs = zeros(Int64,blockSize);
	parallel = false;
	if parallel==false
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
						# ccall((:updateSolution,lib),Void,(Ptr{VankaBlockType},Ptr{Complex128},Ptr{Complex128},Int16,Ptr{Int64},)
									# ,D[:,i],y ,r,convert(Int16,blockSize),Idxs);
						# if norm(x[Idxs] - y[Idxs]) > 1e-14
						# error("Fix updateSolution in C: ",norm(x[Idxs] - y[Idxs]));
						# end
					
					end
				end
			end
		end
	else
		# y[:] = 0.0;
		# void RelaxVankaFacesColor(long long *rowptr , double complex *valA ,long long *colA,
							  # long long *n,long long *nf,long long dim,double complex *x,double complex *b,
							  # float complex *D,long long numit,long long includePressure,long long numCores){
		ccall((:RelaxVankaFacesColor,lib),Void,(Ptr{Int64},Ptr{Complex128},Ptr{Int64},Ptr{Int64}, Ptr{Int64}, Int64,Ptr{Complex128},Ptr{Complex128},Ptr{Complex128},Ptr{VankaBlockType},Int64, Int64, Int64,Int64,),
			AT.colptr,AT.nzval,AT.rowval,n, nf, dim,x,b,y,D,numit,convert(Int64,includePressure),length(x),numCores);
	end
	# if norm(x - y_t) > 1e-14
		# error("Fix RelaxVankaFacesColor in C: ",norm(x - y_t));
	# end
	# println("Diff: ",norm(x-y_t));
	return x;
end