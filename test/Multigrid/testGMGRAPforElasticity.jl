using jInv.Mesh
using Multigrid
using SparseArrays
using LinearAlgebra
println("************************************************* GMG RAP for Elasticity 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [64,64];
Mr = getRegularMesh(domain,n)
mu = 1.0*ones(prod(Mr.n));
lambda = mu;
Ar = GetLinearElasticityOperator(Mr,mu,lambda);
Ar = Ar + 1e-2*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

levels      = 4;
numCores 	= 2; 
maxIter     = 5;
relativeTol = 1e-10;
relaxType   = "SPAI";
relaxParam  = 0.75;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='V';
coarseSolveType = "Julia";
transferOperatorType = "SystemsFacesLinear";

MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,transferOperatorType);

N = size(Ar,2);
b = Ar*rand(N,2); b = b/norm(b);
x = zeros(N,2);
MGsetup(Ar,Mr,MG,Float64,size(b,2),true);
println("****************************** Stand-alone GMG RAP: ******************************")
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.05;
println("****************************** GMG + CG: ******************************")
x[:].=0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.01;


println("************************************************* GMG RAP for Elasticity 3D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0,0.0,1.0];
n = [16,16,12];
Mr = getRegularMesh(domain,n)
mu = 2.0*ones(prod(Mr.n));
lambda = mu;
Ar = GetLinearElasticityOperator(Mr,mu,lambda);
Ar = Ar + 1e-2*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

N = size(Ar,2);
b = Ar*rand(N,2); b = b/norm(b);
x = zeros(N,2);
clear!(MG)
MGsetup(Ar,Mr,MG,Float64,size(b,2),true);
println("****************************** Stand-alone GMG RAP: ******************************")
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.05;
println("****************************** GMG + CG: ******************************")
x[:].=0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.01;





