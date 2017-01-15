using jInv.Mesh
using Multigrid

println("************************************************* Example 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [100,100];
Mr = getRegularMesh(domain,n)
G = getNodalGradientMatrix(Mr);
Ar = G'*G;
Ar = Ar + 1e-4*norm(Ar,1)*speye(size(Ar,2));

levels      = 4;
numCores 	= 8; 
maxIter     = 3;
relativeTol = 1e-10;
relaxType   = "Jac";
relaxParam  = 0.8;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='W';
coarseSolveType = "NoMUMPS";

MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);

N = size(Ar,2);

b = Ar*rand(N,2);
x = zeros(N,2);

MGsetup(Ar,Mr,MG,Float64,size(b,2),true);

println("****************************** Stand-alone GMG: ******************************")
solveMG(MG,b,x,true);

println("****************************** Stand-alone GMG: iterative coarsest ***********************")
coarseSolveType = "BiCGSTAB"
MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);
MGsetup(Ar,Mr,MG,Float64,size(b,2),true);
solveMG(MG,b,x,true);
println("****************************** GMRES preconditioned with GMG: (only one rhs...) ******************************")
x[:] = 0.0
b = vec(b[:,1]);
x = vec(x[:,1]);
relaxType   = "Jac-GMRES";
coarseSolveType = "NoMUMPS"
MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);
MGsetup(Ar,Mr,MG,Float64,size(b,2),true);
solveGMRES_MG(Ar,MG,b,x,true,2)

Ar = 0;
b = 0;
x = 0;
Mr = 0;

copySolver(MG);
destroyCoarsestLU(MG);
clear!(MG)
