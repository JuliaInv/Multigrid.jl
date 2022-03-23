using jInv.Mesh
using Multigrid
using SparseArrays
using LinearAlgebra
using Test
println("************************************************* GMG RAP for Elasticity 2D using Vanka ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
mu = 1.0*ones(prod(Mr.n));
lambda = 10.0.*mu;
#Ar = GetLinearElasticityOperator(Mr,mu,lambda);
Ar = GetLinearElasticityOperatorMixedFormulation(Mr, mu,lambda);
println(norm(Ar - Ar'))	
Ar = Ar + 1e-3*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

levels      = 4;
numCores 	= 2; 
maxIter     = 5;
relativeTol = 1e-10;
relaxType   = "EconVankaFaces";
relaxParam  = 0.75;
relaxPre 	= 1;
relaxPost   = 1;
cycleType   ='V';
coarseSolveType = "Julia";
transferOperatorType = "SystemsFacesMixedLinear";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,transferOperatorType);

N = size(Ar,2);
b = Ar*rand(N); b = b/norm(b);
x = zeros(N);
MGsetup(Ar,Mr,MG,size(b,2),true);
println("****************************** Stand-alone GMG RAP: ******************************")
solveMG(MG,b,x,true);
#println(norm(Ar*x - b))
@test norm(Ar*x - b) < 0.05;
println("****************************** GMG + CG: ******************************")
x[:].=0.0;
solveCG_MG(Ar,MG,b,x,true)
#println(norm(Ar*x - b))
@test norm(Ar*x - b) < 0.01;


println("************************************************* GMG RAP for Elasticity 2D using HybridVanka(Kaczmarz) ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
mu = 1.0*ones(prod(Mr.n));
lambda = 10.0.*mu;
#Ar = GetLinearElasticityOperator(Mr,mu,lambda);
Ar = GetLinearElasticityOperatorMixedFormulation(Mr, mu,lambda);
println(norm(Ar - Ar'))	
Ar = Ar + 1e-3*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

levels      = 4;
numCores 	= 2; 
maxIter     = 5;
relativeTol = 1e-10;
relaxType   = "hybridVankaFacesKaczmarz";
relaxParam  = getHybridKaczmarzParam([4,4],0.9,numCores,2);
relaxPre 	= 1;
relaxPost   = 1;
cycleType   ='V';
coarseSolveType = "Julia";
transferOperatorType = "SystemsFacesMixedLinear";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,transferOperatorType);

N = size(Ar,2);
b = Ar*rand(N); b = b/norm(b);
x = zeros(N);
MGsetup(Ar,Mr,MG,size(b,2),true);
println("****************************** Stand-alone GMG RAP: ******************************")
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.05;
println("****************************** GMG + CG: ******************************")
x[:].=0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.01;



println("************************************************* GMG RAP for Elasticity 3D  using Vanka ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0,0.0,1.0];
# for some reason this test fails to converge at [8,8,8] and succeeds at larger sizes. TODO: check this. 
n = [8,8,8];
# n = [128,128,128];

Mr = getRegularMesh(domain,n)
mu = 1.0*ones(prod(Mr.n));
# lambda = 10.0.*mu;
lambda = 1.0.*mu;
Ar = GetLinearElasticityOperatorMixedFormulation(Mr, mu,lambda);
Ar = Ar + 1e-2*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
MG.relaxType = "VankaFaces";
MG.relaxParam = 0.5;

N = size(Ar,2);
b = Ar*rand(N); b = b/norm(b);
x = zeros(N);
clear!(MG)
MGsetup(Ar,Mr,MG,size(b,1),true);
println("****************************** Stand-alone GMG RAP: ******************************")
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.05;






