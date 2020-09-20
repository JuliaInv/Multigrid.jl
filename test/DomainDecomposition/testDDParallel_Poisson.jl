
using Distributed

if nworkers()==1
	addprocs(2);
end


@everywhere begin
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
end
@everywhere begin
	include(string(pwd(),"/DomainDecomposition/DDPoissonFuncs.jl"));
end


println("************************************************* Dirichlet DomainDecomposition for Poisson 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [128,128];
Mr = getRegularMesh(domain,n)
G = getNodalGradientMatrix(Mr);
Ar = G'*G;
Ar = Ar + 1e-5*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));

NumCells = [4,4];
overlap = [1,1];
n_tup = tuple(n...);
dim = length(n);


numCores 	= 2; 
maxIter     = 10;
relativeTol = 1e-6;

q = randn(prod(n.+1));
x = zeros(size(q)); 
Ainv = getParallelJuliaSolver(Float64,Int64,numCores=numCores,backend=1);

DDparam = getDomainDecompositionParam(Mr,NumCells,overlap,getNodalIndicesOfCell,Ainv);

println("workers are: ",workers())

setupDDParallel(Ar,DDparam,workers());

Prec = r->solveDDParallel(Ar,r,zeros(eltype(q),size(q)),DDparam,workers(),1,0)[1];
x = zeros(size(q)); 

x, flag,rnorm,iter = KrylovMethods.fgmres(getAfun(Ar,zeros(eltype(q),size(q)),4),q,10,tol = 1e-10,maxIter = 2,M = Prec, x = x,out=2,flexible=true);


println(norm(Ar'*x - q)/norm(q))



println("************************************************* Neumann DomainDecomposition for Poisson 2D ******************************************************");

Ctor = DomainDecompositionOperatorConstructor(Mr,getSubParams,getLap,getDirichletMassNodalMesh);
 
setupDDParallel(Ctor,DDparam,workers());

Prec = r->solveDDParallel(Ar,r,zeros(eltype(q),size(q)),DDparam,workers(),1,0)[1];
x = zeros(size(q)); 

x, flag,rnorm,iter = KrylovMethods.fgmres(getAfun(Ar,zeros(eltype(q),size(q)),4),q,10,tol = 1e-10,maxIter = 2,M = Prec, x = x,out=2,flexible=true);
