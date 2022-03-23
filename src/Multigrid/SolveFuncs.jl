export solveMG,solveGMRES_MG,solveBiCGSTAB_MG,solveCG_MG,getAfun

function solveMG(param::MGparam{VAL,IND},b::Array{VAL},x::Array{VAL},verbose::Bool) where {VAL,IND}
#MGType = getMGType(param,b);

param = adjustMemoryForNumRHS(param,size(b,2));
tol = param.relativeTol;
numCores = param.numCores;
oneType = one(VAL);
zeroType = zero(VAL);
maxIter = param.maxOuterIter;
AT = param.As[1];
r = param.memCycle[1].r;
r[:] = b;
if norm(x)==0	
    res = norm(b);
    res_init = res;
else
	SpMatMul(-oneType,AT,x,oneType,r,numCores)#  r -= A'*x;
    res = norm(r);
    res_init = res;
end
iter = 0;
for count = 1:maxIter
	x = recursiveCycle(param,b,x,1);
	SpMatMul(-oneType,AT,x,zeroType,r,numCores); #  r = -A'*x;
	addVectors(oneType,b,r); # r = r + b;
	iter+=1;
	res_prev = res;
	res = norm(r);
	if verbose
		println(string("Cycle ",count," done with relres: ",res/res_init,". Convergence factor: ",res/res_prev));
	end
	if res/res_init < tol 
		break;
	end
end
return x,param,iter;
end
####################################################################################################################
# this function checks if the memory allocated in param fits the nrhs and generates a function for applying the cycle.
# Also - it checks if the hierarchy needs to be transposed or not.
function getMultigridPreconditioner(param::MGparam{VAL,IND},B::Array,verbose::Bool=false) where {VAL,IND}
	n = size(B,1)
	nrhs = size(B,2);
	if hierarchyExists(param)==false
		println("You have to do a setup first.")
	end
	param = adjustMemoryForNumRHS(param,nrhs);
	z = param.memCycle[1].x;
	
	mixed_precision = VAL!=eltype(B);
	MMG = identity;
	if mixed_precision
		bl =  param.memCycle[1].b;
		z2 = zeros(eltype(B),size(B));
		MMG = (b) -> (z[:] .= 0.0;bl[:] .= b; recursiveCycle(param,bl,z,1); z2[:] .= z; return z2;);
	else
		MMG = (b) -> (z[:] .= 0.0;recursiveCycle(param,b,z,1); return z);
	end
	
	return MMG;
end

function getAfun(AT::SparseMatrixCSC{VAL,IND},Az::Array{VAL},numCores::Int64) where {VAL,IND}
	function Afun(z::Array{VAL})
		SpMatMul(AT,z,Az,numCores);
		return Az;
	end
	return Afun;
end


function solveBiCGSTAB_MG(AT::SparseMatrixCSC,param::MGparam{VAL,IND},b::Array,x0::Array,verbose::Bool = false) where {VAL,IND}
	return solveBiCGSTAB_MG(getAfun(AT,zeros(eltype(b),size(b)),param.numCores),param,b,x0,verbose);
end
function solveCG_MG(AT::SparseMatrixCSC,param::MGparam{VAL,IND},b::Array,x0::Array,verbose::Bool = false) where {VAL,IND}
	return solveCG_MG(getAfun(AT,zeros(eltype(b),size(b)),param.numCores),param,b,x0,verbose);
end
function solveGMRES_MG(AT::SparseMatrixCSC,param::MGparam{VAL,IND},b::Array,x0::Array,flexible::Bool,inner::Int64,verbose::Bool = false) where {VAL,IND}
	return solveGMRES_MG(getAfun(AT,zeros(eltype(b),size(b)),param.numCores),param,b,x0,flexible,inner,verbose);
end


function solveBiCGSTAB_MG(Afun::Function,param::MGparam{VAL,IND},b::Array,x0::Array,verbose::Bool = false) where {VAL,IND}
MMG = getMultigridPreconditioner(param,b,verbose);
out= -2;
if verbose
	out = 1;
end
if size(b,2)==1
	b = vec(b);
	x, flag,rnorm,iter = KrylovMethods.bicgstb(Afun,b,tol = param.relativeTol,maxIter = param.maxOuterIter,M1 = MMG,M2 = identity, x = x0,out=out);
else
	x, flag,rnorm,iter = KrylovMethods.blockBiCGSTB(Afun,b,tol = param.relativeTol,maxIter = param.maxOuterIter,M1 = MMG,M2 = identity, x = x0,out=out);
end
nprec = 2*iter*size(b,2) + (flag==-3)*size(b,2);
return x,param,iter,nprec
end



function solveCG_MG(Afun::Function,param::MGparam{VAL,IND},b::Array,x0::Array,verbose::Bool = false) where {VAL,IND}
MMG = getMultigridPreconditioner(param,b,verbose);
out = -2;
if verbose
	out = 1;
end
if size(b,2)==1
	b = vec(b);
	x, flag,rnorm,iter = KrylovMethods.cg(Afun,b,tol = param.relativeTol,maxIter = param.maxOuterIter,M = MMG, x = x0,out=out);
else
	x, flag,rnorm,iter = KrylovMethods.blockCG(Afun,b,tol = param.relativeTol,maxIter = param.maxOuterIter,M = MMG, X = x0,out=out);
end
return x,param,iter;
end



function solveGMRES_MG(Afun::Function,param::MGparam{VAL,IND},b::Array,x0::Array,flexible::Bool,inner::Int64,verbose::Bool = false) where {VAL,IND}
MMG = getMultigridPreconditioner(param,b,verbose);
out = -2;
if verbose
	out = 1;
end
if size(b,2)==1
	b = vec(b);
	x, flag,rnorm,iter,resvec = KrylovMethods.fgmres(Afun,b,inner,tol = param.relativeTol,maxIter = param.maxOuterIter,M = MMG, x = x0,out=out,flexible=flexible);
else
	x, flag,rnorm,iter,resvec = KrylovMethods.blockFGMRES(Afun,b,inner,tol = param.relativeTol,maxIter = param.maxOuterIter,M = MMG, X = x0,out=out,flexible=flexible);
end
return x,param,iter,resvec;
end