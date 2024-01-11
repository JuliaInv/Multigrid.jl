using jInv.Mesh
using Multigrid
using SparseArrays
using LinearAlgebra
using Test
println("************************************************* Pure GMG for VectorLaplacian 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [256,256];
Mr = getRegularMesh(domain,n)
mu = ones(prod(Mr.n));
lambda = 0.0*mu;


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
	Ar = GetVectorLaplacianOperator(Mr,Hparam.mu,Hparam.lambda);
	Ar = Ar + 1e-10*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
	return Ar;
end


levels      = 2;
numCores 	= 2; 
maxIter     = 6;
relativeTol = 1e-10;
relaxType   = "SPAI";
relaxParam  = 0.75;
relaxPre 	= 3;
relaxPost   = 3;
cycleType   ='V';
coarseSolveType = "Julia";
transferOperatorType = "SystemsFacesLinear";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,transferOperatorType);

MGsetup(getMultilevelOperatorConstructor(Hparam,getOperator,restrictParam),Mr,MG,1,true);

Ar = MG.As[1]; # It is actually transposed, but matrix is symmetric.

Arc = MG.As[2];

# MGsetup(Ar,Mr,MG,1,true);
# Arc = MG.As[2];
N = size(Ar,2);
t = rand(N);
t = t .- sum(t)/N;

b = Ar*t; b = b;
x = zeros(N);
println("****************************** Stand-alone GMG: ******************************")
solveMG(MG,b,x,true);
# @test norm(Ar*x - b) < 0.05;
x_true = Ar\b;
x_true = x_true .- sum(x_true)/N;
x = x .- sum(x)/N;
e = x - x_true
println("1")
println(norm(x_true))
using PyPlot
close("all")
r = Ar*e
println("2")
println(norm(r))
r1 = reshape(r[1:(n[1]*(n[2]+1))],(n[1],n[2]+1))
e1 = reshape(e[1:(n[1]*(n[2]+1))],(n[1],n[2]+1))
imshow((e1)); colorbar(); title("error")
figure()
imshow((r1)); colorbar(); title("residual")

# println("****************************** GMG + CG: ******************************")
# x[:].=0.0;
# solveCG_MG(Ar,MG,b,x,true)
# @test norm(Ar*x - b) < 0.01;