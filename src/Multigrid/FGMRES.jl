export getFGMRESmem,FGMRES,FGMRES_relaxation

mutable struct FGMRESmem
	v_prec              ::ArrayTypes ## memory for the result of the preconditioner
	Az					::ArrayTypes ## memory for the result of A*z.
	V    				::ArrayTypes
	Z					::ArrayTypes
end

function resetMem(mem::FGMRESmem)
mem.v_prec[:].=0.0;
mem.Az[:].=0.0;
mem.V[:].=0.0;
mem.Z[:].=0.0;
end

function getFGMRESmem(n::Int64,flexible::Bool,T::Type,k::Int64,m::Int64=1)
## Effectively we do not use v_prec and Az.

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

function FGMRES_relaxation(AT::SparseMatrixCSC,r0::Vector,x0::Vector,inner::Int64,prec::Function,TOL::Float64,
                 verbose::Bool,flexible::Bool,numCores::Int64, mem::FGMRESmem =  getEmptyFGMRESmem())
return FGMRES_relaxation(getAfun(AT,zeros(eltype(r0),size(r0)),numCores),r0,x0,inner,prec,TOL,verbose,flexible,numCores,mem);
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


w = zeros(eltype(r0),0);
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

	BLAS.gemv!('C', constOne, AZ, w,constZero,t);
	
	xi[j] = dot(w,r0);
	
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