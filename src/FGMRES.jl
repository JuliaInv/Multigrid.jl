export getFGMRESmem,FGMRES,FGMRES_relaxation

type FGMRESmem
	V    				::ArrayTypes
	Z					::ArrayTypes
	w					::ArrayTypes
	z					::ArrayTypes
end

function resetMem(mem::FGMRESmem)
mem.V[:]=0.0;
mem.Z[:]=0.0;
mem.w[:]=0.0;
mem.z[:]=0.0;
end

function getFGMRESmem(n::Int64,flexible::Bool,T::Type,k::Int64)
if flexible 
	return FGMRESmem(zeros(T,n,k),zeros(T,n,k),zeros(T,n),zeros(T,n));
else
	return FGMRESmem(zeros(T,n,k),zeros(T,0),zeros(T,n),zeros(T,n));
end
end

function getEmptyFGMRESmem()
return FGMRESmem(zeros(0),zeros(0),zeros(0),zeros(0));
end

function isempty(mem::FGMRESmem)
return size(mem.V,1)==0;
end

function FGMRES(AT::SparseMatrixCSC,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem =  getEmptyFGMRESmem())
function ATfun(alpha,z::Vector,beta,w::Vector)
	w = SpMatMul(alpha,AT,z,beta,w,numCores);
	return w;
end
return FGMRES(ATfun,r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
end


function FGMRES_relaxation(AT::SparseMatrixCSC,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem =  getEmptyFGMRESmem())
function Afun(alpha,z::Vector,beta,w::Vector)
	w = SpMatMul(alpha,AT,z,beta,w,numCores);
	return w;
end
return FGMRES_relaxation(Afun,r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
end


function FGMRES(Afun::Function,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem = getEmptyFGMRESmem())
n = length(r0);
if isempty(mem)
	warn("Allocating memory in GMRES")
	mem = getFGMRESmem(n,flexible,eltype(r0),inner);
else
    resetMem(mem);
	if size(mem.V,2)!= inner
		error("FGMRES: size of Krylov subspace is different than inner");
	end
end

betta = norm(r0);
rnorm0 = betta;

TYPE = eltype(r0);

H = zeros(TYPE,inner+1,inner);
xi = zeros(TYPE,inner+1);
t = zeros(TYPE,inner);
xi[1] = betta;

constOne = one(TYPE);
constZero = zero(TYPE);

w = mem.w;
z = mem.z;
Z = mem.Z;
V = mem.V;

w[:] = r0;
 
BLAS.scal!(n,(1/betta)*constOne,w,1);

rnorms = zeros(inner);

for j = 1:inner
    V[:,j] = w;  # no memory problem with this line....
	z = prec(w,z); # here unfortunately we have an allocation....
	if flexible
		Z[:,j] = z; # no memory problem with this line....
	end
	
	w = Afun(constOne,z,constZero,w); # w = A'*z;# w = SpMatMul(A,z,w,numCores);
	
	
    # Gram Schmidt:
	BLAS.gemv!('C', constOne, V, w,constZero,t);# t = V'*w;
	H[1:inner,j] = t;
	BLAS.gemv!('N', -constOne, V, t,constOne,w); # w = w - V*t  
		
	## modified Gram Schmidt:
    # for i=1:j
        # H[i,j] = dot(vec(V[:,i]),w);
        # w = w - H[i,j]*vec(V[:,i]);
    # end
	
	
    betta = norm(w);
    H[j+1,j] = betta;
    
	BLAS.scal!(n,(1/betta)*constOne,w,1); # w = w*(1/betta);

	# the following 2 lines are equivalent to the 2 next 
    # y = H[1:j+1,1:j]\xi[1:j+1];
    # y = norm(H[1:j+1,1:j]*y - xi[1:j+1])
    Q = qr(H[1:j+1,1:j],thin=false)[1];
	rnorms[j] = abs(Q[1,end]*xi[1]);
      if verbose
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
y = pinv(H)*xi;
# norm(H*y - xi)
if flexible
	BLAS.gemv!('N', constOne, Z, y,constZero,w); # w = Z*y  #This is the correction that corresponds to the residual. 
else
	BLAS.gemv!('N', constOne, V, y, constZero, w);
	z = prec(w,z);
	w[:] = z;
end
addVectors(constOne,w,x0);
return x0,rnorms; # w is the correction vector;
end


function FGMRES_relaxation(AT::Function,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem = getEmptyFGMRESmem())

n = length(r0);
if isempty(mem)
	warn("Allocating memory in GMRES")
	mem = getFGMRESmem(n,flexible,eltype(r0),inner);
else
    resetMem(mem);
	if size(mem.V,2)!= inner
		error("FGMRES: size of Krylov subspace is different than inner");
	end
end

rnorm0 = norm(r0);

TYPE = eltype(r0);

H = zeros(TYPE,inner,inner);
xi = zeros(TYPE,inner);
t = zeros(TYPE,inner);
D = zeros(TYPE,inner,inner); # normalization matrix.

constOne = one(TYPE);
constZero = zero(TYPE);

w = mem.w;
z = mem.z;
Z = mem.Z;
AZ = mem.V;

w[:] = r0;

rnorms = zeros(inner);

for j = 1:inner
    z = prec(w,z); # here unfortunately we have an allocation....
	
	Z[:,j] = z; # no memory problem with this line....
	
	w = AT(constOne,z,constZero,w); # w = A'*z;# w = SpMatMul(A,z,w,numCores);
	
	AZ[:,j] = w;  # no memory problem with this line....

	
	xi[j] = dot(w,r0);	
	
	BLAS.gemv!('C', constOne, AZ, w,constZero,t);
	
	H[:,j] = t;
	H[j,:] = t';
	H = 0.5*H + 0.5*H';
	t = pinv(H)*xi;
	
	rnorms[j] = sqrt(real(dot(t,H*t)-2.0*dot(t,xi)+rnorm0^2));

    if verbose
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
BLAS.gemv!('N', constOne, Z, t,constZero,w); # w = Z*y  #This is the correction that corresponds to the residual. 
addVectors(constOne,w,x0);

return x0,rnorms; # w is the correction vector;
end