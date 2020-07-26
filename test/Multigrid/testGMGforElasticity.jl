using jInv.Mesh
using Multigrid
using SparseArrays
using LinearAlgebra
println("************************************************* Pure GMG for Elasticity 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
mu = 1.0*ones(prod(Mr.n));
lambda = mu;


mutable struct ElasticityParam
	Mesh   			:: RegularMesh;
	lambda          :: Array{Float64};
	mu				:: Array{Float64}
	MixedFormulation:: Bool
end

Hparam = ElasticityParam(Mr,lambda,mu,false);

function restrictParam(mesh_fine,mesh_coarse,param_fine,level)
	mu_c 	 	= restrictCellCenteredVariables(param_fine.mu,mesh_fine.n);
	lambda_c 	= restrictCellCenteredVariables(param_fine.lambda,mesh_fine.n);
	return ElasticityParam(mesh_coarse,lambda_c,mu_c,param_fine.MixedFormulation);
end

function getOperator(Mr,Hparam)
	Ar = GetLinearElasticityOperator(Mr,Hparam.mu,Hparam.lambda);
	Ar = Ar + 1e-3*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
	return Ar;
end


levels      = 5;
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

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,transferOperatorType);

MGsetup(getMultilevelOperatorConstructor(Hparam,getOperator,restrictParam),Mr,MG,1,true);
Ar = MG.As[1]; # It is actually transposed, but matrix is symmetric.

N = size(Ar,2);
b = Ar*rand(N); b = b/norm(b);
x = zeros(N);
println("****************************** Stand-alone GMG: ******************************")
solveMG(MG,b,x,true);
@test norm(Ar*x - b) < 0.05;
println("****************************** GMG + CG: ******************************")
x[:].=0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.01;


# println("************************************************* GMG for Elasticity 3D ******************************************************");

# domain = [0.0, 1.0, 0.0, 1.0,0.0,1.0];
# n = [16,16,12];
# Mr = getRegularMesh(domain,n)
# mu = 2.0*ones(prod(Mr.n));
# lambda = mu;
# Ar = GetLinearElasticityOperator(Mr,mu,lambda);
# Ar = Ar + 1e-2*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

# N = size(Ar,2);
# b = Ar*rand(N,2); b = b/norm(b);
# x = zeros(N,2);
# clear!(MG)
# MGsetup(Ar,Mr,MG,size(b,2),true);
# println("****************************** Stand-alone GMG RAP: ******************************")
# solveMG(MG,b,x,true);
# @test norm(Ar*x - b) < 0.05;
# println("****************************** GMG + CG: ******************************")
# x[:].=0.0;
# solveCG_MG(Ar,MG,b,x,true)
# @test norm(Ar*x - b) < 0.01;





