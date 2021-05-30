function recursiveCycle(param::MGparam{VAL,IND},b::Array{VAL},x::Array{VAL},level::Int64) where {VAL,IND}

# println(string("Starting level ", level));

gmresTol = 1e-5;

n = size(b,1);
nrhs = size(b,2);
numCores = param.numCores;
As = param.As;
nlevels = length(As);

if level==nlevels # This actually does not need to happen unless one level only is used (i.e. exact solver).
	r = param.memCycle[level].r;
	r[:] = b;
	x = solveCoarsest(param,r,x);
	return x;
end

Ps = param.Ps;
Rs = param.Rs;
AT = As[level];

oneType = one(VAL);
zeroType = zero(VAL);
r = param.memCycle[level].r;

r[:] = b;
if norm(x)>0.0	
   	SpMatMul(-oneType,AT,x,oneType,r,numCores)#  r -= A'*x;
end
D = param.relaxPrecs[level];
MM = identity;
Afun = identity;
if param.relaxType=="Jac-GMRES"
	y = param.memRelax[level].v_prec;
	MM(xx::Array{VAL}) = (return y .= D.*xx);
end

PT = Ps[level];
RT = Rs[level];

npresmth  = param.relaxPre(level);
npostsmth = param.relaxPost(level);

if param.relaxType=="Jac-GMRES"
    Afun = getAfun(AT,param.memRelax[level].Az,numCores)
	x = FGMRES_relaxation(Afun,r,x,npresmth,MM,gmresTol,false,numCores,param.memRelax[level])[1];
elseif param.relaxType == "VankaFaces"
	x = RelaxVankaFacesColor(AT,x,b,D,npresmth,numCores,param.Meshes[level],param.transferOperatorType=="SystemsFacesMixedLinear",FULL_VANKA);
elseif param.relaxType == "EconVankaFaces"
	x = RelaxVankaFacesColor(AT,x,b,D,npresmth,numCores,param.Meshes[level],param.transferOperatorType=="SystemsFacesMixedLinear",ECON_VANKA);
elseif param.relaxType=="hybridVankaFacesKaczmarz"
	x = RelaxHybridVanka(param.relaxPrecs[level], AT,x,b,npresmth,param.relaxPrecs[level].numCores,param.Meshes[level],
								param.transferOperatorType=="SystemsFacesMixedLinear",KACMARZ_VANKA);
else
	x = relax(AT,r,x,b,D,npresmth,numCores);
end

SpMatMul(-oneType,AT,x,zeroType,r,numCores); #  r = -A'*x;

addVectors(oneType,b,r); # r = r + b;


xc = param.memCycle[level+1].x;
xc[:] .= 0.0;
bc = param.memCycle[level+1].b;
bc = SpMatMul(RT,r,bc,numCores); 
if level==nlevels-1
	# println("solving coarsest");
	xc = solveCoarsest(param,bc,xc);
else
	Ac = As[level+1];
    if param.cycleType == 'K'
		yzK = param.memKcycle[level].v_prec;
		AfunK = getAfun(Ac,param.memKcycle[level].Az,numCores);
		MMG(x) = (yzK .= 0.0; recursiveCycle(param,x,yzK,level+1)); # x does not change...
		xc = FGMRES_relaxation(AfunK,bc,xc,2,MMG,gmresTol,false,numCores,param.memKcycle[level])[1];
    else
		xc = recursiveCycle(param,bc,xc,level+1);# xc changes, bc does not change
		if param.cycleType=='W'
            xc = recursiveCycle(param,bc,xc,level+1);
		elseif param.cycleType=='F'
            param.cycleType='V';
            xc = recursiveCycle(param,bc,xc,level+1);
			param.cycleType='F';
        end
    end
end


x = SpMatMul(oneType,PT,xc,oneType,x,numCores); # x += PT'*xc;

r[:] = b;
SpMatMul(-oneType,AT,x,oneType,r,numCores); #  r -= A'*x;
if param.relaxType=="Jac-GMRES"
	Afun = getAfun(AT,param.memRelax[level].Az,numCores)
	x = FGMRES_relaxation(Afun,r,x,npostsmth,MM,gmresTol,false,numCores,param.memRelax[level])[1];
elseif param.relaxType == "VankaFaces"
	x = RelaxVankaFacesColor(AT,x,b,D,npostsmth,numCores,param.Meshes[level],param.transferOperatorType=="SystemsFacesMixedLinear",FULL_VANKA);
elseif param.relaxType == "EconVankaFaces"
	x = RelaxVankaFacesColor(AT,x,b,D,npostsmth,numCores,param.Meshes[level],param.transferOperatorType=="SystemsFacesMixedLinear", ECON_VANKA);
elseif param.relaxType=="hybridVankaFacesKaczmarz"
	x = RelaxHybridVanka(param.relaxPrecs[level], AT,x,b,npostsmth,param.relaxPrecs[level].numCores,param.Meshes[level],
								param.transferOperatorType=="SystemsFacesMixedLinear",KACMARZ_VANKA);
else
	x = relax(AT,r,x,b,D,npostsmth,numCores);
end

return x
end



function relax(AT::SparseMatrixCSC{VAL,IND},r::Array{VAL},x::Array{VAL},b::Array{VAL},d::Vector{VAL},numit::Int64,numCores::Int64) where {VAL,IND}
# x is actually the output... x and is being changed in the iterations.
# r does not end up accurate becasue we do not update it in the last iteration.
oneType = one(eltype(r));
zeroType = zero(eltype(r));
# nr0 = vecnorm(r);
for i=1:numit-1
	x .+= d.*r;
	SpMatMul(-oneType,AT,x,zeroType,r,numCores) # r = -A'*x
	addVectors(oneType,b,r); # r = r + b;
	# println("Reduced: ", vecnorm(r)/nr0)
end
x .+= d.*r;
return x
end


function solveCoarsest(param::MGparam{VAL,IND},b::Array{VAL},x::Array{VAL},doTranspose::Int64=0) where {VAL,IND}

if isa(param.LU,DomainDecompositionParam)
	AT = param.As[end];
	(x,param.LU) = solveDDSerial(AT,b,x,param.LU,1,doTranspose);
	return x;
elseif isa(param.LU,AbstractSolver)
	AT = param.As[end];
	(x,param.LU) = solveLinearSystem!(AT,b,x,param.LU);
	return x;
end

if param.coarseSolveType == "MUMPS"
	applyMUMPS(param.LU,b,xt,param.doTranspose);
elseif param.coarseSolveType == "GMRES"
	AT = param.As[end];
	maxIter = 1;
	tol = 0.01;
	out= -2;
	Afun = getAfun(AT,zeros(eltype(b),size(b)),param.numCores);
	d = param.LU;
	y = zeros(eltype(b),size(b));
	M2(xx::Array{VAL}) = (y .= d.*xx; return y);
	if size(b,2)==1
		b = vec(b);
		x[:] .= 0.0;
		x, flag,rnorm,iter = KrylovMethods.fgmres(Afun,b,10,tol = tol,maxIter = 1,M = M2,out=out,x=x);
	else
		x[:] .= 0.0;
		x, flag,rnorm,iter = KrylovMethods.blockFGMRES(Afun,b,10,tol = tol,maxIter = 1,M = M2,out=out,X=x);
	end
elseif param.coarseSolveType == "VankaFaces"
	AT = param.As[end];
	inner = 20;
	Afun = getAfun(AT,zeros(eltype(b),size(b)),param.numCores);
	out= -2;
	x = KrylovMethods.fgmres(Afun,b,inner;tol = 0.8,maxIter = 5,M = param.LU,x = x,flexible = true,out = out)[1];
	# println("Coarsest norm is: ",norm(Afun(y) - b)/norm(b));
else
	z = param.LU\b;	
	x[:] = z;
end
return x;
end








