using jInv.Mesh
using Multigrid
using SparseArrays
using LinearAlgebra
println("************************************************* GMG for Elasticity 2D using Vanka ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
mu = 1.0*ones(prod(Mr.n));
lambda = 1.0.*mu;

mutable struct ElasticityParam
	Mesh   			:: RegularMesh;
	lambda          :: Array{Float64};
	mu				:: Array{Float64}
	MixedFormulation:: Bool
end

Hparam = ElasticityParam(Mr,lambda,mu,true);

function restrictParam(mesh_fine,mesh_coarse,param_fine,level)
	mu_c 	 	= restrictCellCenteredVariables(param_fine.mu,mesh_fine.n);
	lambda_c 	= restrictCellCenteredVariables(param_fine.lambda,mesh_fine.n);
	return ElasticityParam(mesh_coarse,lambda_c,mu_c,true);
end

# function GetVectorLapOperatorMixedFormulation(M::RegularMesh, mu::Vector,lambda::Vector)	
	# vecGRAD,~,Div,nf, = getDifferentialOperators(M,2);
	# Mu     			  = getTensorMassMatrix(M,mu[:],saveAvMat=false)[1];
	# Rho	   			  = getFaceMassMatrix(M,1e-5*mu, saveMat = false, avN2C = avN2C_Nearest);
	# A 				  = vecGRAD'*Mu*vecGRAD + Rho;
	# A 				  = spdiagm(0=>ones(size(Rho,1)));
	# C 				  = -(spdiagm(0=>0.0./(lambda[:])));
	# Div.nzval .*= -1.0;
	# H = [A  Div' ; -Div  -C;];
	# return H;						
# end

# function GetVectorLapOperator(M::RegularMesh, mu::Vector,lambda::Vector)	
	# vecGRAD,~,Div,nf, = getDifferentialOperators(M,2);
	# Mu     			  = getTensorMassMatrix(M,mu[:],saveAvMat=false)[1];
	# Rho	   			  = getFaceMassMatrix(M,1e-5*mu, saveMat = false, avN2C = avN2C_Nearest);
	# A 				  = vecGRAD'*Mu*vecGRAD + Rho;
	# return A;						
# end





function getOperator(Mr,Hparam)
	# Ar = GetVectorLapOperator(Mr, Hparam.mu,Hparam.lambda);
	# Ar = GetVectorLapOperatorMixedFormulation(Mr, Hparam.mu,Hparam.lambda);
	Ar = GetLinearElasticityOperatorMixedFormulation(Mr, Hparam.mu,Hparam.lambda);
	Ar = Ar + 1e-3*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
	return Ar;
end

levels      = 5;
numCores 	= 2; 
maxIter     = 5;
relativeTol = 1e-10;
relaxType   = "VankaFaces";
# relaxType   = "Jac";
relaxParam  = 0.5;
relaxPre 	= 1;
relaxPost   = 1;
cycleType   ='V';
coarseSolveType = "Julia";
transferOperatorType = "SystemsFacesMixedLinear";

MG = getMGparam(Float64,Int64,levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0,transferOperatorType);

MGsetup(getMultilevelOperatorConstructor(Hparam,getOperator,restrictParam),Mr,MG,1,true);
Ar = MG.As[1]; # It is actually transposed, but matrix is symmetric. 
# MGsetup(Ar,Mr,MG,1,true);


N = size(Ar,2);
sol = randn(N);
sol = sol/norm(sol);
sol = sol .- sum(sol)./N;
b = Ar*sol; b = b/norm(b);
x = zeros(N);

println("****************************** Stand-alone GMG: ******************************")
x, = solveMG(MG,copy(b),x,true);
@test norm(Ar*x - b) < 0.05;


println("****************************** GMG + CG: ******************************")
x[:].=0.0;
solveCG_MG(Ar,MG,b,x,true)
@test norm(Ar*x - b) < 0.01;




println("TODO: see why the code below doesn't work.")
clear!(MG)

# println("************************************************* GMG for Elasticity 3D  using Vanka ******************************************************");

# domain = [0.0, 1.0, 0.0, 1.0,0.0,1.0];
# n = [16,16,16];
# Mr = getRegularMesh(domain,n)
# mu = 1.0*ones(prod(Mr.n));
# lambda = 10.0.*mu;

# Hparam = ElasticityParam(Mr,lambda,mu,true);

# MGsetup(getMultilevelOperatorConstructor(Hparam,getOperator,restrictParam),Mr,MG,size(b,1),true);
# Ar = MG.As[1]; # It is actually transposed, but matrix is symmetric.
# N = size(Ar,2);
# b = Ar*rand(N); b = b/norm(b);
# x = zeros(N);


# println("****************************** Stand-alone 3D GMG: ******************************")
# solveMG(MG,b,x,true);
# @test norm(Ar*x - b) < 0.05;
# println("****************************** 3D GMG + CG: ******************************")
# x[:].=0.0;
# solveCG_MG(Ar,MG,b,x,true)
# @test norm(Ar*x - b) < 0.01;





