export getFGMRESmem,FGMRES,FGMRES_relaxation

type FGMRESmem
	v_prec              ::ArrayTypes ## memory for the result of the preconditioner
	Az					::ArrayTypes ## memory for the result of A*z.
	V    				::ArrayTypes
	Z					::ArrayTypes
end

function resetMem(mem::FGMRESmem)
mem.v_prec[:]=0.0;
mem.Az[:]=0.0;
mem.V[:]=0.0;
mem.Z[:]=0.0;
end

function getFGMRESmem(n::Int64,flexible::Bool,T::Type,k::Int64,m::Int64=1)
v_prec = zeros(T,0);
Az = zeros(T,0);
if m==1
	Az = zeros(T,n);
	v_prec = zeros(T,n);
else
	Az = zeros(T,n,m);
	v_prec = zeros(T,n,m);
end

if flexible 
	return FGMRESmem(v_prec,Az,zeros(T,n,k*m),zeros(T,n,k*m));
else
	return FGMRESmem(v_prec,Az,zeros(T,n,k*m),zeros(T,0));
end
end


function getEmptyFGMRESmem()
return FGMRESmem(zeros(0),zeros(0),zeros(0),zeros(0));
end
import Base.isempty
function isempty(mem::FGMRESmem)
return size(mem.V,1)==0;
end

function FGMRES(AT::SparseMatrixCSC,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem =  getEmptyFGMRESmem())
Az = zeros(eltype(r0),size(r0));				 
function Afun(z::Vector)
	SpMatMul(AT,z,Az,numCores);
	return Az;
end
return FGMRES(Afun,r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
end


function FGMRES_relaxation(AT::SparseMatrixCSC,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem =  getEmptyFGMRESmem())
Az = zeros(eltype(r0),size(r0));				 
function Afun(z::Vector)
	SpMatMul(AT,z,Az,numCores);
	return Az;
end
return FGMRES_relaxation(Afun,r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
end


function FGMRES(Afun::Function,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem = getEmptyFGMRESmem())
n = length(r0);
if isempty(mem)
	# warn("Allocating memory in GMRES")
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
y = copy(t);
xi[1] = betta;

constOne = one(TYPE);
constZero = zero(TYPE);


Z = mem.Z;
V = mem.V;


 
BLAS.scal!(n,(1/betta)*constOne,r0,1); # w = w./betta

rnorms = zeros(inner);
w = 0;
for j = 1:inner
	if j==1
		V[:,j] = r0;  # no memory problem with this line....
		z = prec(r0)
	else
		V[:,j] = w;  # no memory problem with this line....
		z = prec(w)
	end
	
	if flexible
		Z[:,j] = z; # no memory problem with this line....
	end
	w = Afun(z);
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
        # println(string("Inner iter ",j,": FGMRES: residual norm: ",rnorms[j],", relres: ",rnorms[j]/rnorm0,", factor: ",rnorms[j]/resprev));
	end
    if rnorms[j] < TOL
        rnorms = rnorms[1:j];
        break;
    end
end
y[:] = pinv(H)*xi;
# norm(H*y - xi)
if flexible
	BLAS.gemv!('N', constOne, Z, y,constZero,w); # w = Z*y  #This is the correction that corresponds to the residual. 
else
	BLAS.gemv!('N', constOne, V, y, constZero, w);
	z = prec(w);
	w = z;
end
addVectors(constOne,w,x0);

return x0,rnorms; # w is the correction vector;
end


function FGMRES_relaxation(Afun::Function,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem = getEmptyFGMRESmem())

n = length(r0);
if isempty(mem)
	warn("Allocating memory in GMRES")
	mem = getFGMRESmem(n,true,eltype(r0),inner);
else
    resetMem(mem);
	if size(mem.V,2)!= inner
		error("FGMRES_relaxation: size of Krylov subspace is different than inner");
	end
end

rnorm0 = norm(r0);

TYPE = eltype(r0);

H = zeros(TYPE,inner,inner);
xi = zeros(TYPE,inner);
t = zeros(TYPE,inner);

constOne = one(TYPE);
constZero = zero(TYPE);


Z = mem.Z;
AZ = mem.V;

# w[:] = r0;
w = 0;
rnorms = zeros(inner);

for j = 1:inner
	if j==1
		z = prec(r0); # here unfortunately we have an allocation....
	else
		z = prec(w);
	end
	Z[:,j] = z; # no memory problem with this line....
	
	w = Afun(z); # w = A'*z;# w = SpMatMul(A,z,w,numCores);
	AZ[:,j] = w;  # no memory problem with this line....

	
	xi[j] = dot(w,r0);	
	
	BLAS.gemv!('C', constOne, AZ, w,constZero,t);
	
	H[:,j] = t;
	H[j,:] = t';
	H = 0.5*H + 0.5*H';
	t[:] = pinv(H)*xi;
	
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