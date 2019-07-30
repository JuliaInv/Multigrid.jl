export getBlockFGMRESmem,BlockFGMRES_relaxation,BlockFGMRESmem,isempty
export BlockFGMRES


function BlockFGMRES(AT::SparseMatrixCSC,r0::ArrayTypes,x0::ArrayTypes,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem =  getEmptyFGMRESmem())
Az = zeros(eltype(r0),size(r0));				 
function Afun(z::ArrayTypes)
	SpMatMul(AT,z,Az,numCores);
	return Az;
end
return BlockFGMRES(Afun,r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
end


function BlockFGMRES(Afun::Function,R0::ArrayTypes,X0::ArrayTypes,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem = getEmptyFGMRESmem())
(n,m) = size(R0);
if isempty(mem)
	warn("Allocating memory in Block GMRES")	
	mem = getFGMRESmem(n,flexible,eltype(R0),inner,false);
else
    resetMem(mem);
	if size(mem.V,2)!= inner*m
		error("BlockFGMRES_relaxation: size of Krylov subspace is different than inner times #rhs");
	end
end

Zbig = mem.Z;
Vbig = mem.V;


rnorm0 = vecnorm(R0);
Q = qrfact!(R0); Betta = Q[:R]; R0 = full(Q[:Q],thin=true); # this code is equivalent to (W,Betta) = qr(R0);
W = 0;
Z = 0;


TYPE = eltype(R0);
H = zeros(TYPE,(inner+1)*m,inner*m);
xi = zeros(TYPE,(inner+1)*m,m);
T = zeros(TYPE,inner*m,m);
xi[1:m,:] = Betta;

constOne = one(TYPE);
constZero = zero(TYPE);

rnorms = zeros(inner);


for j = 1:inner

	colSet_j = (((j-1)*m)+1):j*m
	if j==1
		Vbig[:,colSet_j] = R0;  # no memory problem with this line....
		Z = prec(R0); 
	else
		Vbig[:,colSet_j] = W;  # no memory problem with this line....
		Z = prec(W);
	end
	if flexible
		Zbig[:,colSet_j] = Z;
	end
	
	
	W = Afun(Z); # w = A'*z;

	BLAS.gemm!('C','N', constOne, Vbig, W,constZero,T); # t = V'*w;
	H[1:(inner*m),colSet_j] = T;
	BLAS.gemm!('N','N', -constOne, Vbig, T,constOne,W); # w = w - V*t  
	
	Q = qrfact!(W); Betta = Q[:R]; W = full(Q[:Q],thin=true); # this code is equivalent to (W,Betta) = qr(W);
	
	H[((j*m)+1):(j+1)*m,colSet_j] = Betta;
	
	y = H[1:(j+1)*m,1:j*m]\xi[1:(j+1)*m,:];
    rnorms[j] = vecnorm(H[1:(j+1)*m,1:j*m]*y - xi[1:(j+1)*m,:])
	if verbose
		if j==1
			resprev = rnorm0;
		else
			resprev = rnorms[j-1];
		end
        # println(string("Inner iter ",j,": BFGMRES: residual norm: ",rnorms[j],", relres: ",rnorms[j]/rnorm0,", factor: ",rnorms[j]/resprev));
    end
	if rnorms[j] < TOL
        rnorms = rnorms[1:j];
        break;
    end
end
y = pinv(H)*xi;
# y = H\xi;

if flexible
	BLAS.gemm!('N','N', constOne, Zbig, y,constZero,W); # w = Z*y  #This is the correction that corresponds to the residual.
else
	# W = Vbig*y;
	BLAS.gemm!('N','N', constOne, Vbig, y, constZero, W);
	Z = prec(W);
	W = Z;
end
addVectors(constOne,W,X0);
return X0,rnorms,mem; # w is the correction vector;
end



mutable struct BlockFGMRESmem
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

import Base.isempty
function isempty(mem::BlockFGMRESmem)
return length(mem.AZbig)==0;
end

function getBlockFGMRESmem(n::Int64,m::Int64,flexible::Bool,T::Type,inner::Int64)
if m <= 1
	error("Block GMRES is used for 1 rhs. Use GMRES");
end
if flexible
	Zbig = Array(Array{T,2},inner);
	for kk=1:inner
		Zbig[kk] = zeros(T,n,m);
	end
	AZbig = Array(Array{T,2},inner);
	for kk=1:inner
		AZbig[kk] = zeros(T,n,m);
	end
else
	Zbig = Array(Array{T,2},0);
	# Zbig[1] = zeros(T,n,m);
	AZbig = Array(Array{T,2},inner);
	for kk=1:inner
		AZbig[kk] = zeros(T,n,m);
	end
end
return BlockFGMRESmem(Zbig,AZbig);
end

function BlockFGMRES_relaxation(AT::SparseMatrixCSC,r0::ArrayTypes,x0::ArrayTypes,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::BlockFGMRESmem =  getEmptyBlockFGMRESmem())
Az = zeros(eltype(r0),size(r0));				 
function Afun(z::ArrayTypes)
	SpMatMul(AT,z,Az,numCores);
	return Az;
end
return BlockFGMRES_relaxation(Afun,r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
end


# function BlockFGMRES_relaxation(Afun::Function,R0::ArrayTypes,X0::ArrayTypes,inner::Int64,prec::Function,TOL::Float64,
                 # verbose::Bool,flexible::Bool,numCores::Int64, mem::BlockFGMRESmem = getEmptyBlockFGMRESmem())
# warning("This method is not numerically stable");
# n = size(R0,1);
# m = size(R0,2);
# if isempty(mem)
	# warn("Allocating memory in Block GMRES")	
	# mem = getBlockFGMRESmem(n,m,flexible,eltype(R0),inner);
# else
    # resetMem(mem);
	# if length(mem.AZbig)!= inner 
		# error("BlockFGMRES_relaxation: size of Krylov subspace is different than inner");
	# end
# end
# Zbig = mem.Zbig; # if flexible
# AZbig = mem.AZbig;

# rnorm0 = vecnorm(R0);

# TYPE = eltype(R0);

# H = zeros(TYPE,inner*m,inner*m);
# Xi = zeros(TYPE,inner*m,m);
# T = zeros(TYPE,inner*m,m);
# Told = copy(T);
# S = zeros(TYPE,m,m);

# constOne = one(TYPE);
# constZero = zero(TYPE);

# rnorms = zeros(inner);
# Z = 0;
# W = 0;

# for j = 1:inner
	# # println("---------------");
	# colSet_j = (((j-1)*m)+1):j*m
	# if j==1
		# Z = prec(R0);
	# else
		# Z = prec(AZbig[j-1]);
	# end
	# if flexible
		# Zbig[j][:] = Z;
	# end
	# W = Afun(Z); # w = A'*z;
	# AZbig[j][:] = W;
	
	# BLAS.gemm!('C','N', constOne, AZbig[j], R0,constZero,S); # t = V'*w;
	# Xi[colSet_j,:] = S;    
	# # Xi[colSet_j,:] = dot(AZbig[j],R0)
	# for jj = 1:j
		# BLAS.gemm!('C','N', constOne, AZbig[jj], AZbig[j],constZero,S);
		# # S = dot( AZbig[jj], AZbig[j]);
		# if j==jj
			# S = 0.5*S+0.5*S';
		# end
		# colSet_jj = (((jj-1)*m)+1):jj*m
		# T[colSet_jj,:] = S;
	# end

	# H[:,colSet_j] = T;
	# H[colSet_j,:] = T';
	# if j>1
		# ee = eig(H)[1];
		# ee = ee[ee.!=0.0];
		# if ee!=[]
			# if minimum(ee)/maximum(ee) < 1e-14
				# if verbose
					# println("Breaking GMRES primitive due to lack of accuracy in LS solver");
				# end
				# # T = Told;
				# rnorms = rnorms[1:j-1];
				# break;
			# end
		# end
	# end
		

	# T = pinv(H)*Xi;
	
	# # println(real(trace(T'*H*T-2.0*T'*Xi))+rnorm0^2)
	# rnorms[j] = sqrt(abs(real(trace(T'*H*T-2.0*T'*Xi))+rnorm0^2));
	# if verbose
		# ee = eig(H)[1];
		# ee = ee[ee.!=0.0];
		# # println("Condition Number H: ",minimum(ee)/maximum(ee));
		# if j==1
			# resprev = rnorm0;
		# else
			# resprev = rnorms[j-1];
		# end
        # println(string("Inner iter ",j,": FGMRES: residual norm: ",rnorms[j],", relres: ",rnorms[j]/rnorm0,", factor: ",rnorms[j]/resprev));
    # end
    # if rnorms[j] < TOL
        # rnorms = rnorms[1:j];
        # break;
    # end
# end
# if flexible
	# for jj = 1:length(rnorms)
		# colSet_jj = (((jj-1)*m)+1):jj*m
		# BLAS.gemm!('N','N', constOne, Zbig[jj], T[colSet_jj,:],constOne,X0);
		# # addVectors(T[jj],Zbig[jj],X0);
	# end
# else
	# jj = 1;
	# colSet_jj = (((jj-1)*m)+1):jj*m
	# BLAS.gemm!('N','N', constOne, R0, T[colSet_jj,:],constZero,Z);
	# for jj = 2:length(rnorms)
		# colSet_jj = (((jj-1)*m)+1):jj*m
		# BLAS.gemm!('N','N', constOne, AZbig[jj-1], T[colSet_jj,:],constOne,Z);
	# end
	# AZbig[1] = prec(Z);
	# addVectors(constOne,AZbig[1],X0);
# end

# return X0,rnorms,mem; # w is the correction vector;
# end