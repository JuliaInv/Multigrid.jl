using jInv.Mesh
using Multigrid
using SparseArrays
using LinearAlgebra
using Test
println("************************************************* GMG RAP for Poisson 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
G = getNodalGradientMatrix(Mr);
Ar = G'*G;
Ar = Ar + 1e-4*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

levels      = 4;
numCores 	= 8; 
maxIter     = 5;
relativeTol = 1e-10;
relaxType   = "Jac-GMRES";
relaxParam  = 0.75;
relaxPre 	= 1;
relaxPost   = 1;
cycleType   ='V';
coarseSolveType = "NoMUMPS";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);

N = size(Ar,2);

b = Ar*rand(N,2);
b = b/norm(b);
x = zeros(N,2);  


MGsetup(Ar,Mr,MG,size(b,2),true);

println("****************************** Stand-alone GMG RAP for Poisson: ******************************")
x = solveMG(MG,b,x,true)[1];
println("*******************Outside: ",norm(Ar*x - b)/norm(b))
@test norm(Ar*x - b) < 0.005;
println("****************************** Stand-alone GMG RAP : iterative coarsest ***********************")
coarseSolveType = "GMRES"
MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);
MGsetup(Ar,Mr,MG,size(b,2),true);
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.001;
println("****************************** GMRES preconditioned with GMG, Jac-GMRES relaxation: ******************************")
x[:] .= 0.0
relaxType   = "Jac-GMRES";
coarseSolveType = "NoMUMPS"
MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);
MGsetup(Ar,Mr,MG,size(b,2),true);
x = solveGMRES_MG(Ar,MG,b,x,true,10,true)[1];
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Outside: ",norm(Ar*x - b)/norm(b))
@test norm(Ar*x - b) < 0.001;

println("****************************** 3D Poisson ******************************")

domain = [0.0, 1.0, 0.0, 1.0,0.0,1.0];
n = [32,32,16];
Mr = getRegularMesh(domain,n)
G = getNodalGradientMatrix(Mr);
Ar = G'*G;
Ar = Ar + 1e-4*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0);

N = size(Ar,2);

b = Ar*rand(N,2);
b = b/norm(b);
x = zeros(N,2);

MGsetup(Ar,Mr,MG,size(b,2),true);

println("****************************** Stand-alone 3D GMG RAP Poisson: ******************************")
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.01;


Ar = 0;
b = 0;
x = 0;
Mr = 0;
copySolver(MG);
destroyCoarsestLU(MG);
Multigrid.clear!(MG)
MG = 0;