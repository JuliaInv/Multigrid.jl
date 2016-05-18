export getBlockFGMRESmem,BlockFGMRES_relaxation,BlockFGMRESmem,isempty

type BlockFGMRESmem
	Zbig    			::Array{ArrayTypes}
	AZbig				::Array{ArrayTypes}
end

function resetMem(mem::BlockFGMRESmem)
for k = 1:length(mem.AZbig)
	mem.AZbig[k][:] = 0.0;
end
for k = 1:length(mem.Zbig)
	mem.Zbig[k][:] = 0.0;
end
end

function getEmptyBlockFGMRESmem(n::Int64,m::Int64,flexible::Bool,T::Type,k::Int64)
Zbig  = Array(Array{T,2},0);
AZbig = Array(Array{T,2},0);
return BlockFGMRESmem(Zbig,AZbig);
end

function isempty(mem::BlockFGMRESmem)
return length(mem.Zbig)==0;
end

function getBlockFGMRESmem(n::Int64,m::Int64,flexible::Bool,T::Type,inner::Int64)
if m <= 1
	error("Block GMRES is used for 1 rhs. Use GMRES");
end
if flexible
	# if m > 1
		Zbig = Array(Array{T,2},inner);
		for kk=1:inner
			Zbig[kk] = zeros(T,n,m);
		end
		AZbig = Array(Array{T,2},inner);
		for kk=1:inner
			AZbig[kk] = zeros(T,n,m);
		end
	# else
		# Zbig = Array(Array{T,1},inner);
		# for kk=1:inner
			# Zbig[kk] = zeros(T,n);
		# end
		# AZbig = Array(Array{T,1},inner);
		# for kk=1:inner
			# AZbig[kk] = zeros(T,n);
		# end
	# end
else
	# if m > 1
		Zbig = Array(Array{T,2},1);
		Zbig[1] = zeros(T,n,m);
		AZbig = Array(Array{T,2},inner);
		for kk=1:inner
			AZbig[kk] = zeros(T,n,m);
		end
	# else
		# Zbig = Array(Array{T,1},1);
		# Zbig[1] = zeros(T,n);
		# AZbig = Array(Array{T,1},inner);
		# for kk=1:inner
			# AZbig[kk] = zeros(T,n);
		# end
	# end
end
return BlockFGMRESmem(Zbig,AZbig);
end

function BlockFGMRES_relaxation(AT::SparseMatrixCSC,r0::ArrayTypes,x0::ArrayTypes,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::BlockFGMRESmem =  getEmptyBlockFGMRESmem())
function ATfun(alpha,Z::ArrayTypes,beta,W::ArrayTypes)
	W = SpMatMul(alpha,AT,Z,beta,W,numCores);
	return W;
end
return BlockFGMRES_relaxation(ATfun,r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
end


function BlockFGMRES_relaxation(Afun::Function,R0::ArrayTypes,X0::ArrayTypes,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::BlockFGMRESmem = getEmptyBlockFGMRESmem())

n = size(R0,1);
m = size(R0,2);
if isempty(mem)
	warn("Allocating memory in Block GMRES")	
	mem = getBlockFGMRESmem(n,m,flexible,eltype(R0),inner);
else
    # resetMem(mem);
	if length(mem.AZbig)!= inner 
		error("FGMRES: size of Krylov subspace is different than inner");
	end
end
Zbig = mem.Zbig; # if flexible
AZbig = mem.AZbig;

rnorm0 = vecnorm(R0);

TYPE = eltype(R0);

H = zeros(TYPE,inner*m,inner*m);
Xi = zeros(TYPE,inner*m,m);
T = zeros(TYPE,inner*m,m);
Told = copy(T);
S = zeros(TYPE,m,m);

constOne = one(TYPE);
constZero = zero(TYPE);

rnorms = zeros(inner);

for j = 1:inner
	# println("---------------");
	colSet_j = (((j-1)*m)+1):j*m
	if flexible 
		if j==1
			Zbig[j] = prec(R0,Zbig[j]);
		else
			Zbig[j] = prec(AZbig[j-1],Zbig[j]);
		end
		Afun(constOne,Zbig[j],constZero,AZbig[j]); # w = A'*z;
	else
		if j==1
			Zbig[1] = prec(R0,Zbig[1]);
		else
			Zbig[1] = prec(AZbig[j-1],Zbig[1]);
		end
		Afun(constOne,Zbig[1],constZero,AZbig[j]); # w = A'*z;
	end
	
	BLAS.gemm!('C','N', constOne, AZbig[j], R0,constZero,S); # t = V'*w;
	Xi[colSet_j,:] = S;    
	# Xi[colSet_j,:] = dot(AZbig[j],R0)
	for jj = 1:j
		BLAS.gemm!('C','N', constOne, AZbig[jj], AZbig[j],constZero,S);
		# S = dot( AZbig[jj], AZbig[j]);
		if j==jj
			S = 0.5*S+0.5*S';
		end
		colSet_jj = (((jj-1)*m)+1):jj*m
		T[colSet_jj,:] = S;
	end

	H[:,colSet_j] = T;
	H[colSet_j,:] = T';
	if j>1
		ee = eig(H)[1];
		ee = ee[ee.!=0.0];
		if ee!=[]
			if minimum(ee)/maximum(ee) < 1e-14
				if verbose
					println("Breaking GMRES primitive due to lack of accuracy in LS solver");
				end
				T = Told;
				rnorms = rnorms[1:j-1];
				break;
			end
		end
	end
		
		
	T = pinv(H)*Xi;
	# println(real(trace(T'*H*T-2.0*T'*Xi))+rnorm0^2)
	rnorms[j] = sqrt(abs(real(trace(T'*H*T-2.0*T'*Xi))+rnorm0^2));
	if verbose
		ee = eig(H)[1];
		ee = ee[ee.!=0.0];
		println("Condition Number H: ",minimum(ee)/maximum(ee));
		if j==1
			resprev = rnorm0;
		else
			resprev = rnorms[j-1];
		end
        println(string("Inner iter ",j,": FGMRES: residual norm: ",rnorms[j],", relres: ",rnorms[j]/rnorm0,", factor: ",rnorms[j]/resprev));
    end
    if rnorms[j] < TOL
        rnorms = rnorms[1:j];
        break;
    end
end
if flexible
	for jj = 1:length(rnorms)
		colSet_jj = (((jj-1)*m)+1):jj*m
		BLAS.gemm!('N','N', constOne, Zbig[jj], T[colSet_jj,:],constOne,X0);
		# addVectors(T[jj],Zbig[jj],X0);
	end
else
	jj = 1;
	colSet_jj = (((jj-1)*m)+1):jj*m
	BLAS.gemm!('N','N', constOne, R0, T[colSet_jj,:],constZero,Zbig[1]);
	for jj = 2:length(rnorms)
		colSet_jj = (((jj-1)*m)+1):jj*m
		BLAS.gemm!('N','N', constOne, AZbig[jj-1], T[colSet_jj,:],constOne,Zbig[1]);
	end
	AZbig[1] = prec(Zbig[1],AZbig[1]);
	addVectors(constOne,AZbig[1],X0);
end

return X0,rnorms,mem; # w is the correction vector;
end

