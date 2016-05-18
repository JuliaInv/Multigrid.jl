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
innerIter   = 0;
relativeTol = 1e-10;
relaxType   = "SPAI";
relaxParam  = 1.0;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='V';
coarseSolveType = "NoMUMPS";

MG = getMGparam(levels,numCores,maxIter,innerIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);

N = size(Ar,2);

b = Ar*rand(N,3);
x = zeros(N,3);
SA_AMGsetup(Ar,MG,Float64,true,size(b,2),true);

println("****************************** Stand-alone AMG: ******************************")
solveMG(MG,b,x,true);

x[:] = 0.0;
println("****************************** CG preconditioned with AMG: ******************************")
tic()
solveCG_MG(Ar,MG,b,x,true)
toc()

MG.innerIter = 3;
println("****************************** GMRES preconditioned with AMG: (only one rhs...) ******************************")
x[:] = 0.0;
b = vec(b[:,1]);
x = zeros(N);
tic()
solveGMRES_MG(Ar,MG,b,x,true)
toc()

Ar = 0;
b = 0;
x = 0;
Mr = 0;