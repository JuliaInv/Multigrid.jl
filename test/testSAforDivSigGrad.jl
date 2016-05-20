using jInv.Mesh
using DivSigGrad
using Multigrid

println("************************************************* Example 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [50,50];
Mr = getRegularMesh(domain,n)
m  = exp(randn(prod(n)));
Ar = getDivSigGradMatrix(vec(m),Mr);
Ar = Ar + 1e-8*norm(Ar,1)*speye(size(Ar,2));

levels      = 3;
numCores 	= 8; 
maxIter     = 3;
relativeTol = 1e-10;
relaxType   = "SPAI";
relaxParam  = 1.0;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='V';
coarseSolveType = "NoMUMPS";

MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);

N = size(Ar,2);

b = Ar*rand(N,3);
x = zeros(N,3);
SA_AMGsetup(Ar,MG,Float64,true,size(b,2),true);

println("****************************** Stand-alone AMG: ******************************")
solveMG(MG,b,x,true);

x[:] = 0.0;
println("****************************** CG preconditioned with AMG: ******************************")
solveCG_MG(Ar,MG,b,x,true)

println("****************************** CG preconditioned with BiCGSTAB: ******************************")
solveBiCGSTAB_MG(Ar,MG,b,x,true);

println("****************************** transposeHierarchy *********************************");
transposeHierarchy(MG,true);


println("****************************** GMRES preconditioned with AMG: (only one rhs...) ******************************")
x[:] = 0.0;
b1 = vec(b[:,1]);
x1 = zeros(N);
solveGMRES_MG(Ar,MG,b1,x1,true,2)

x1 = zeros(N);
MG.maxOuterIter = 10;
solveBiCGSTAB_MG(Ar,MG,b1,x1,true)


MG = getMGparam(levels,numCores,maxIter,relativeTol,"Jac-GMRES",relaxParam,relaxPre,relaxPost,'K',coarseSolveType);
SA_AMGsetup(Ar,MG,Float64,true,size(b,2),true);
x[:] = 0.0;
solveGMRES_MG(Ar,MG,b,x,true,2)

Ar = 0;
b = 0;
x = 0;
Mr = 0;