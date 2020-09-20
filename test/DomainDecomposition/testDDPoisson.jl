


using jInv.Mesh
using jInv.LinearSolvers
using Multigrid.ParallelJuliaSolver
using Multigrid.DomainDecomposition
using Multigrid
using KrylovMethods

using SparseArrays
using LinearAlgebra
using Test
using jInv.Mesh;

include("DDPoissonFuncs.jl");



println("************************************************* Dirichlet DomainDecomposition for Poisson 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
G = getNodalGradientMatrix(Mr);
Ar = G'*G;
Ar = Ar + 1e-5*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

NumCells = [8,8];
overlap = [1,1];
n_tup = tuple(n...);
dim = length(n);


numCores 	= 2; 
maxIter     = 10;
relativeTol = 1e-6;

q = randn(prod(n.+1));
x = zeros(size(q)); 
Ainv = getParallelJuliaSolver(Float64,Int64,numCores=numCores,backend=1);
# Ainv = getJuliaSolver();
DDparam = getDomainDecompositionParam(Mr,NumCells,overlap,getNodalIndicesOfCell,Ainv);

(~,DDparam) = solveLinearSystem!(Ar,q,x,DDparam)

println(norm(Ar'*x - q)/norm(q))



println("************************************************* Neumann DomainDecomposition for Poisson 2D ******************************************************");




Ctor = DomainDecompositionOperatorConstructor(Mr,getSubParams,getLap,getDirichletMassNodalMesh);
 
setupDDSerial(Ctor,DDparam);

Prec = r->solveDDSerial(Ar,r,zeros(eltype(q),size(q)),DDparam,1,0)[1];
x = zeros(size(q)); 

x, flag,rnorm,iter = KrylovMethods.fgmres(getAfun(Ar,zeros(eltype(q),size(q)),4),q,10,tol = 1e-10,maxIter = 2,M = Prec, x = x,out=2,flexible=true);


# x = zeros(size(q)); 

# x, flag,rnorm,iter = KrylovMethods.bicgstb(getAfun(Ar,zeros(eltype(q),size(q)),4), q, tol = 1e-10,maxIter = 25,M1 = Prec, x = x,out=2)		



# println("****************************** DomainDecomposition for Poisson 3D ******************************")

# domain = [0.0, 1.0, 0.0, 1.0,0.0,1.0];
# n = [64,64,32];
# Mr = getRegularMesh(domain,n)
# G = getNodalGradientMatrix(Mr);
# Ar = G'*G;
# Ar = Ar + 1e-4*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
# N = size(Ar,2);

# q = Ar*rand(N);
# q = q/norm(q);
# x = zeros(N);


# numCores 	= 2; 
# maxIter     = 10;
# relativeTol = 1e-6;


# NumCells = [4,4,2];
# overlap = [2,2,2];

# Ainv = getParallelJuliaSolver(Float64,Int64,numCores=numCores,backend=1);
# # Ainv = getJuliaSolver();
# DDparam = getDomainDecompositionParam(Mr,NumCells,overlap,getNodalIndicesOfCell,Ainv);

# (~,DDparam) = solveLinearSystem!(Ar,q,x,DDparam)
# # x = solveLinearSystem!(Ar,q,x,DDparam)[1];

# println(norm(Ar'*x - q)/norm(q))




###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################














# # println("****************************  Original equation:    ******************************")				

# # SHT = SH';
# NumCells = [8,4];
# overlap = [12,12];
# # DDfactors = setupDD(SHT,Minv,getFacesStaggeredIndicesOfCellNoPressure,NumCells,overlap,0.0);
# # x = solveDD(HT,Minv.n,q,zeros(Complex128,size(q)),getFacesStaggeredIndicesOfCellNoPressure,NumCells,overlap,DDfactors,30);
# # println(norm(H*x - q)/norm(q))


# # println("DIRICHLET: ")
# # Prec = r->solveDD(HT,Minv.n,r,zeros(Complex128,size(q)),getFacesStaggeredIndicesOfCellNoPressure,NumCells,overlap,DDfactors,1);
# # x, flag,rnorm,iter = KrylovMethods.bicgstb(H,q,tol = 1e-6,maxIter = 50,M1 = Prec,M2 = identity, x = zeros(eltype(q),size(q)),out=2);
# # println(norm(H*x - q)/norm(q));

# # Prec = r->solveDDJac(HT,Minv.n,r,zeros(Complex128,size(q)),getFacesStaggeredIndicesOfCellNoPressure,NumCells,overlap,DDfactors,1);
# # x, flag,rnorm,iter = KrylovMethods.bicgstb(H,q,tol = 1e-6,maxIter = 50,M1 = Prec,M2 = identity, x = zeros(eltype(q),size(q)),out=2);
# # println(norm(H*x - q)/norm(q));


 
# # LU = lufact(SH);
# # Mexact(x) = LU\x;
# # x, flag,rnorm,iter = KrylovMethods.bicgstb(H,q,tol = 1e-6,maxIter = 50,M1 = Mexact,M2 = identity, x = zeros(eltype(q),size(q)),out=2);

# # x[:] = 0.0;
# # for k=1:50
	# # r = q - H*x;
	# # println(k,": ",norm(r)/norm(q));
	# # x = x + Mexact(r);
# # end
# ############################################################################################################################
# ############################################ Geometric Test ################################################################
# ############################################################################################################################

# println("ABL: ")

# ElasticHelmholtzConstructor = DomainDecompositionOperatorConstructor(Hparam,getSubParams,GetElasticHelmholtzOperator);
# DDfactors = setupDD(ElasticHelmholtzConstructor,Minv,identity,NumCells,overlap,0.0);

# Prec = r->solveDD(HT,Minv.n,r,zeros(Complex128,size(q)),getFacesStaggeredIndicesOfCellNoPressure,NumCells,overlap,DDfactors,1);
# x, flag,rnorm,iter = KrylovMethods.bicgstb(H,q,tol = 1e-6,maxIter = 50,M1 = Prec,M2 = identity, x = zeros(eltype(q),size(q)),out=2);
# println(norm(H*x - q)/norm(q));


# # Prec = r->solveDDRB(HT,Minv.n,r,zeros(Complex128,size(q)),getFacesStaggeredIndicesOfCellNoPressure,NumCells,overlap,DDfactors,1);
# # x, flag,rnorm,iter = KrylovMethods.bicgstb(H,q,tol = 1e-6,maxIter = 50,M1 = Prec,M2 = identity, x = zeros(eltype(q),size(q)),out=2);
# # println(norm(H*x - q)/norm(q));


# # Prec = r->solveDDJac(HT,Minv.n,r,zeros(Complex128,size(q)),getFacesStaggeredIndicesOfCellNoPressure,NumCells,overlap,DDfactors,1);
# # x, flag,rnorm,iter = KrylovMethods.bicgstb(H,q,tol = 1e-6,maxIter = 50,M1 = Prec,M2 = identity, x = zeros(eltype(q),size(q)),out=2);
# # println(norm(H*x - q)/norm(q));

























