using jInv.Mesh
using Multigrid
using Multigrid.DomainDecomposition
using SparseArrays
using LinearAlgebra

speye = (n) -> SparseMatrixCSC(1.0I,n,n);

domain = [0.0, 1.0, 0.0, 1.0];
n = [64,64];
Mr = getRegularMesh(domain,n)
numCores = 4;
innerKatzIter = 5;
inner = 5;
nrhs = 2;

println("************************************************* Hybrid Keczmarz for DivSigGrad 2D ******************************************************");

m  = exp.(randn(prod(n)));
Ar = getNodalDivSigGradMatrix(Mr,vec(m));
Ar = Ar + 2e-1*opnorm(Ar,1)*speye(size(Ar,2));
Ar = convert(SparseMatrixCSC{Float64,Int64},Ar)

N = size(Ar,2);
b = Ar*rand(Float64,N,nrhs); b = b/norm(b);
x = zeros(Float64,N,nrhs);


HKparam = getHybridKaczmarz(Float64,Int64, Ar,Mr,[4,4], getNodalIndicesOfCell,0.8,numCores,innerKatzIter);
prec = getHybridKaczmarzPrecond(HKparam,Ar,nrhs);
x = FGMRES_relaxation(Ar,copy(b),x,inner,prec,1e-5*norm(b),true,numCores)[1]
println("Norm is: ",norm(Ar*x - b))

println("************************************************* Hybrid Keczmarz for Elasticity 2D ******************************************************");

mu = 2.0*ones(prod(Mr.n));
lambda = mu;
Ar = GetLinearElasticityOperator(Mr,mu,lambda);
Ar = Ar + 2e-1*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
N = size(Ar,2);
b = Ar*rand(N,nrhs); b = b/norm(b);
x = zeros(N,nrhs);

HKparam = getHybridKaczmarz(Float64,Int64, Ar,Mr,[4,4], getFacesStaggeredIndicesOfCellNoPressure,0.8,numCores,innerKatzIter);
prec = getHybridKaczmarzPrecond(HKparam,Ar,nrhs);
x = FGMRES_relaxation(Ar,copy(b),x,inner,prec,1e-5*norm(b),true,numCores)[1]
println("Norm is: ",norm(Ar*x - b))