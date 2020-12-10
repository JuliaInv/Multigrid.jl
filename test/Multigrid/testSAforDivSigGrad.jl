using LinearAlgebra
using SparseArrays
using jInv.Mesh
using Multigrid
using Test

println("************************************************* AMG Setup For DivSigGrad ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [50,50];
Mr = getRegularMesh(domain,n)
m  = exp.(randn(prod(n)));
Ar = getNodalDivSigGradMatrix(Mr,vec(m));
Ar = Ar + 1e-8*norm(Ar,1)*sparse(1.0I, size(Ar,2),size(Ar,2));

levels      = 3;
numCores 	= 2; 
maxIter     = 5;
relativeTol = 1e-4;
relaxType   = "SPAI";
relaxParam  = 1.0;
relaxPre 	= 1;
relaxPost   = 1;
cycleType   ='V';
coarseSolveType = "Julia";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);

N = size(Ar,2);

b = Ar*rand(N,3);
b = b/norm(b);
x = zeros(N,3);
SA_AMGsetup(Ar,MG,true,size(b,2),true);

println("****************************** Stand-alone SA-AMG: ******************************")
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.01;


println("****************************** CG preconditioned with SA-AMG: ******************************")
x[:] .= 0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.005;


println("****************************** BiCGSTAB preconditioned with SA-AMG: ******************************")
x[:] .= 0.0;
solveBiCGSTAB_MG(Ar,MG,b,x,true);
@test norm(Ar*x - b) < 0.005;

println("****************************** transposeHierarchy *********************************");
transposeHierarchy(MG,true);


println("****************************** replaceMatrixHierarchy *********************************");
replaceMatrixInHierarchy(MG,Ar,true);

println("****************************** GMRES preconditioned with SA-AMG: (only one rhs...) ******************************")
x[:] .= 0.0;
b1 = vec(b[:,1]);
x1 = zeros(N);
MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,"GMRES");
SA_AMGsetup(Ar,MG,true,1,true);
solveGMRES_MG(Ar,MG,b1,x1,true,2,true)
@test norm(Ar*x1 - b1) < 0.001;
println("****************************** Classical AMG *********************************");
x1 = zeros(N);
println("Stand-alone Classical AMG one rhs:")
MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);
ClassicalAMGsetup(Ar,MG,true,size(b1,2),true);
solveMG(MG,b1,x1,true);
println("CG preconditioned with C-AMG in 3D:")
x1[:] .= 0.0;
solveCG_MG(Ar,MG,b1,x1,true)
@test norm(Ar*x1 - b1) < 0.005;

println("****************************** GMRES preconditioned with SA-AMG: (multiple rhs...) ******************************")

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,"Jac-GMRES",relaxParam,relaxPre,relaxPost,'K',coarseSolveType);
SA_AMGsetup(Ar,MG,true,size(b,2),true);
x[:] .= 0.0;
solveGMRES_MG(Ar,MG,b,x,true,2,true)

Ar = 0;
b = 0;
x = 0;
Mr = 0;




println("************************************************* Example AMG 3D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
n = [32,32,16];
Mr = getRegularMesh(domain,n)
m  = exp.(randn(prod(n)));
Ar = getNodalDivSigGradMatrix(Mr,vec(m));
Ar = Ar + 1e-6*norm(Ar,1)*sparse(1.0I, size(Ar,2),size(Ar,2));

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,"Julia");

N = size(Ar,2);
b = Ar*rand(N,3);
b = b/norm(b);

x = zeros(N,3);
println("Stand-alone SA-AMG:")
SA_AMGsetup(Ar,MG,true,size(b,2),true);
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.005;
println("CG preconditioned with SA-AMG in 3D:")
x[:] .= 0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.005;

x = zeros(N,3);
println("Stand-alone Classical AMG:")
MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,"GMRES");
ClassicalAMGsetup(Ar,MG,true,size(b,2),true);
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.005;
println("CG preconditioned with C-AMG in 3D:")
x[:] .= 0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.005;

Ar = 0;
b = 0;
x = 0;
Mr = 0;


