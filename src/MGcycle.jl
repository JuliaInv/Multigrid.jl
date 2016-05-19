
function recursiveCycle(param::MGparam,b::ArrayTypes,x::ArrayTypes,level::Int64)

# println(string("Starting level ", level));

gmresTol = 1e-5;

n = size(b,1);
nrhs = size(b,2);
numCores = param.numCores;
As = param.As;
nlevels = length(As);


if level==nlevels # This actually does not need to happen unless one level only is used (i.e. exact solver).
	x = solveCoarsest(param,b,x,param.doTranspose);
	return x;
end

Ps = param.Ps;
Rs = param.Rs;
AT = As[level];

const oneType = one(eltype(b));
const zeroType = zero(eltype(b));
r = param.memCycle[level].r;

r[:] = b;
if vecnorm(x)>0.0	
   	SpMatMul(-oneType,AT,x,oneType,r,numCores)#  r -= A'*x;
end
D = param.relaxPrecs[level];
MM(xx::ArrayTypes,y::ArrayTypes) = (SpMatMul(D,xx,y,numCores);return y;);

PT = Ps[level];
RT = Rs[level];

npresmth  = param.relaxPre(level);
npostsmth = param.relaxPost(level);

if param.relaxType=="Jac-GMRES"
	
	if nrhs == 1
		x = FGMRES_relaxation(AT,r,x,npresmth,MM,gmresTol,false,false,numCores,param.memRelax[level])[1];
	else
		x = BlockFGMRES_relaxation(AT,r,x,npresmth,MM,gmresTol,false,false,numCores, param.memRelax[level])[1];
	end
else
	x = relax(AT,r,x,b,D,npresmth,numCores);
end

SpMatMul(-oneType,AT,x,zeroType,r,numCores); #  r = -A'*x;
addVectors(oneType,b,r); # r = r + b;

xc = param.memCycle[level+1].x;
xc[:] = 0.0;
bc = param.memCycle[level+1].b;
bc = SpMatMul(RT,r,bc,numCores); 
if level==nlevels-1
	# println("solving coarsest");
	xc = solveCoarsest(param,bc,xc,param.doTranspose);
else
	Ac = As[level+1];
    if param.cycleType == 'K'
		MMG(x,y) = (y[:] = 0.0; recursiveCycle(param,x,y,level+1)); # x does not change...
		if nrhs==1
			xc = FGMRES_relaxation(Ac,bc,xc,2,MMG,gmresTol,false,true,numCores,param.memKcycle[level+1])[1];
		else
			xc = BlockFGMRES_primitive(Ac,bc,xc,2,MMG,gmresTol,false,true,numCores,param.memKcycle[level+1])[1];
		end
    else
		
		# println("before Coarse solve:",vecnorm(Ac'*xc-bc));
		
		xc = recursiveCycle(param,bc,xc,level+1);# xc changes, bc does not change
        # println("After Coarse solve 1:",vecnorm(Ac'*xc-bc));
		if param.cycleType=='W'
            xc = recursiveCycle(param,bc,xc,level+1);
        elseif param.cycleType=='F'
            param.cycleType='V';
            xc = recursiveCycle(param,bc,xc,level+1);
			param.cycleType='F';
        end
		# println("After Coarse solve 2:",vecnorm(Ac'*xc-bc));
    end
end


# Base.A_mul_B!(oneType,P,xc,oneType,x); # x += P*xc;

x = SpMatMul(oneType,PT,xc,oneType,x,numCores); # x += PT'*xc;

r[:] = b;
SpMatMul(-oneType,AT,x,oneType,r,numCores); #  r -= A'*x;
if param.relaxType=="Jac-GMRES"
	if nrhs == 1
		x = FGMRES_relaxation(AT,r,x,npostsmth,MM,gmresTol,false,false,numCores,param.memRelax[level])[1];
	else
		x = BlockFGMRES_relaxation(AT,r,x,npostsmth,MM,gmresTol,false,false,numCores, param.memRelax[level])[1];
	end
else
	x = relax(AT,r,x,b,D,npostsmth,numCores);
end

return x
end



function relax(AT::SparseMatrixCSC,r::ArrayTypes,x::ArrayTypes,b::ArrayTypes,D::SparseMatrixCSC,numit::Int64,numCores::Int64)
# x is actually the output... x and is being changed in the iterations.
# r does not end up accurate becasue we do not update it in the last iteration.
const oneType = one(eltype(r));
const zeroType = zero(eltype(r));
# nr0 = vecnorm(r);
# println(nr0)
for i=1:numit	
	SpMatMul(oneType,D,r,oneType,x,numCores); # x = x + D'*r
	SpMatMul(-oneType,AT,x,zeroType,r,numCores) # r = -A'*x
	addVectors(oneType,b,r); # r = r + b;
	# println("Reduced: ", vecnorm(r)/nr0)
end

return x
end


function solveCoarsest(param::MGparam,b::ArrayTypes,x::ArrayTypes,doTranspose::Int64=0)
if param.coarseSolveType == "MUMPS"
	applyMUMPS(param.LU,b,x,doTranspose);
else
	if doTranspose==1
		error(" MULTIGRID: solveCoarsest(): If doTranspose ==1 then it's better to use MUMPS ");
	end
	x = param.LU\b;
end
return x;
end

