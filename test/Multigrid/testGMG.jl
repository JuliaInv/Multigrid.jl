using jInv.Mesh;
using Multigrid
using LinearAlgebra
using SparseArrays
println("***************** test GMG discretization ***************************")
# function getNodalLaplacianMatrixDirichlet(Minv::RegularMesh)
# newDomain = copy(Minv.domain);
# newDomain[1] -= Minv.h[1];
# newDomain[2] += Minv.h[1];
# newDomain[3] -= Minv.h[2];
# newDomain[4] += Minv.h[2];
# MinvPadded = getRegularMesh(newDomain,Minv.n+2);
# mask = zeros(Bool,tuple(MinvPadded.n+1...));
# mask[2:end-1,2:end-1] = true;
# A = getNodalLaplacianMatrix(MinvPadded);
# A = A[mask[:],mask[:]];
# return A;
# end

m = ones(129,129);

Minv = getRegularMesh([0.0,1.0,0.0,1.0],collect(size(m)).-1);

levels      = 4;
numCores 	= 2; 
maxIter     = 5;
relativeTol = 1e-2;
relaxType   = "Jac";
relaxParam  = 0.8;
relaxPre 	= 1;
relaxPost   = 1;
cycleType   ='V';
coarseSolveType = "NoMUMPS";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0)

## TEST DIRICHLET
# L = getNodalLaplacianMatrixDirichlet(Minv);
# s = zeros(tuple(Minv.n+1...));
# s[2:end-1,2:end-1] = rand(tuple(Minv.n-1...));
# s = s[:];
# MGsetup(getNodalLaplacianMatrixDirichlet,Minv,MG,1,true)
# MGsetup(L,Minv,MG,1,true)

## TEST NEUMANN
L = getNodalLaplacianMatrix(Minv);
s = rand(prod(Minv.n.+1));
b = L*s;b = b/norm(b);
x0 = 0.0*s;

MGsetup(getMultilevelOperatorConstructor([],getNodalLaplacianMatrix,[]),Minv,MG,1,true);
solveMG(MG,b,x0,true);
@test norm(L*x0 - b) < 0.005;
xn = getCellCenteredGrid(Minv);
sig = 3*xn[:,1].*(1.0.-xn[:,1]) + 2*xn[:,2].*(1.0.-xn[:,2]);
sig = sig[:];
# sig[:]=1.0;

# Geometric multigrid only works with a sigma average that does nearest neighbor at the boundaries.

Ar = getNodalDivSigGradMatrix(Minv,sig);
b = Ar*s;b = b/norm(b)
MGsetup(Ar,Minv,MG,1,true);
x0 = 0.0*s;
solveMG(MG,b,x0,true);
@test norm(Ar*x0 - b) < 0.005;


restrictSigma(mesh_fine,mesh_coarse,param_fine,level) = restrictCellCenteredVariables(param_fine,mesh_fine.n);
MGsetup(getMultilevelOperatorConstructor(sig,getNodalDivSigGradMatrix,restrictSigma),Minv,MG,1,true);
x0 = 0.0*s;
solveMG(MG,b,x0,true);
@test norm(Ar*x0 - b) < 0.005;
println("DONE!")