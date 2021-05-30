using jInv.Mesh
using jInv.LinearSolvers
using Multigrid.ParallelJuliaSolver
using SparseArrays
using LinearAlgebra
using Test

speye = (n) -> SparseMatrixCSC(1.0I,n,n);


println("===  Test Julia wrapper with symmetric (real) matrix ====");
domain = [0.0, 1.0, 0.0, 1.0];
n      = [20,23];

Mr     = getRegularMesh(domain,n)
G      = getNodalGradientMatrix(Mr);
m      = sparse(Diagonal(exp.(randn(size(G,1)))));
Ar     = G'*m*G;
Ar     = Ar + 1e-1*norm(Ar,1)*speye(size(Ar,2));
N      = size(Ar,2);
b      = Ar*rand(N);
bs     = Ar*rand(N,5);
Bs     = (convert(Array{Float64,1}, b),convert(Array{Float64,2}, bs));


LU = getParallelJuliaSolver(Float64,Int64,numCores=1,backend=1);

for j=1:length(Bs)
	print("nrhs=$(size(Bs[j],2)), issparse(rhs)=$(issparse(Bs[j])) : ")
	t = @elapsed local x,  = solveLinearSystem(Ar,Bs[j],LU);
	@test norm(Ar*x-Bs[j],Inf)/norm(Bs[j],Inf) < 1e-8
	#println("V0: took ",t);
	println("LU solve time: ",LU.solveTime);
	LU.solveTime=0.0;
end

clear!(LU);
LU = getParallelJuliaSolver(Float64,Int64,numCores=1,backend=2);

for j=1:length(Bs)
	print("nrhs=$(size(Bs[j],2)), issparse(rhs)=$(issparse(Bs[j])) : ")
	t = @elapsed local x,  = solveLinearSystem(Ar,Bs[j],LU);
	@test norm(Ar*x-Bs[j],Inf)/norm(Bs[j],Inf) < 1e-8
	#println("V1: took ",t);
	println("LU solve time: ",LU.solveTime);
	LU.solveTime=0.0;
end

LU = getParallelJuliaSolver(Float64,Int64,numCores=2,backend=3);

for j=1:length(Bs)
	print("nrhs=$(size(Bs[j],2)), issparse(rhs)=$(issparse(Bs[j])) : ")
	t = @elapsed local x,  = solveLinearSystem(Ar,Bs[j],LU);
	@test norm(Ar*x-Bs[j],Inf)/norm(Bs[j],Inf) < 1e-8
	#println("V1: took ",t);
	println("LU solve time: ",LU.solveTime);
	LU.solveTime=0.0;
end

Bs     = (convert(Array{Float32,1}, b),convert(Array{Float32,2}, bs));
LU = getParallelJuliaSolver(Float32,UInt32,numCores=2,backend=3);

for j=1:length(Bs)
	print("nrhs=$(size(Bs[j],2)), issparse(rhs)=$(issparse(Bs[j])) : ")
	t = @elapsed local x,  = solveLinearSystem(Ar,Bs[j],LU);
	@test norm(Ar*x-Bs[j],Inf)/norm(Bs[j],Inf) < 1e-4
	#println("V1: took ",t);
	println("LU solve time: ",LU.solveTime);
	LU.solveTime=0.0;
end

println("===  Test Julia wrapper with shifted (complex) Laplacian ====");
Ar     = G'*m*G + (1+1im)*speye(size(Ar,2));
b      = Ar*rand(ComplexF64,N);
bs     = Ar*rand(ComplexF64,N,5);
Bs     = (b,bs);

LU = copySolver(LU);
clear!(LU);

LU = getParallelJuliaSolver(ComplexF64,Int64,numCores=2,backend=3);

for j=1:length(Bs)
	print("nrhs=$(size(Bs[j],2)), issparse(rhs)=$(issparse(Bs[j])) : ")
	local x,  = solveLinearSystem(Ar,Bs[j],LU);
	@test norm(Ar*x-Bs[j],Inf)/norm(Bs[j],Inf) < 1e-10
	print("\n")
end
println("")
Bs     = (convert(Array{ComplexF32,1}, b),convert(Array{ComplexF32,2}, bs));
LU = getParallelJuliaSolver(ComplexF32,UInt32,numCores=2,backend=3);
for j=1:length(Bs)
	print("nrhs=$(size(Bs[j],2)), issparse(rhs)=$(issparse(Bs[j])) : ")
	local x,  = solveLinearSystem(Ar,Bs[j],LU);
	@test norm(Ar*x-Bs[j],Inf)/norm(Bs[j],Inf) < 1e-4
	print("\n")
end
println("")

println("===  Test Parellel Julia Wrapper: nonsymmetric matrices ====");
n = 50
sNonSym  = getParallelJuliaSolver(Float64,Int64,numCores=1,backend=1);
sNonSym  = copySolver(sNonSym);
A = sprandn(n,n,5/n) + 10*speye(n);
B = randn(Float64,n)
Bs = randn(Float64,n,5)

X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-5

X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-5

X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-5

X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-5

X, = solveLinearSystem(A,Bs,sNonSym,1);
@test norm(A'*X - Bs)/norm(Bs) < 1e-5


println("===  Testing parallel backend with transposed  ===")
sNonSym  = getParallelJuliaSolver(Float64,Int64,numCores=2,backend=3);
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-8
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-8
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-8
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-8
X, = solveLinearSystem(A,Bs,sNonSym,1);
@test norm(A'*X - Bs)/norm(Bs) < 1e-8

println("===  Testing parallel backend with transposed  ===")
sNonSym  = getParallelJuliaSolver(Float32,UInt32,numCores=2,backend=3);
B = randn(Float32,n)
Bs = randn(Float32,n,5)
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,Bs,sNonSym,1);
@test norm(A'*X - Bs)/norm(Bs) < 1e-4

println("===  Testing parallel backend with complex transposed 64bit  ===")
sNonSym  = getParallelJuliaSolver(ComplexF64,Int64,numCores=2,backend=3);
B = randn(ComplexF64,n)
Bs = randn(ComplexF64,n,5)
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-5
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-5
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-5
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-5
X, = solveLinearSystem(A,Bs,sNonSym,1);
@test norm(A'*X - Bs)/norm(Bs) < 1e-5

println("===  Testing parallel backend with complex transposed 32 bit ===")
sNonSym  = getParallelJuliaSolver(ComplexF32,UInt32,numCores=2,backend=3);
B = randn(ComplexF32,n)
Bs = randn(ComplexF32,n,5)
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,B,sNonSym,1);
@test norm(A'*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,B,sNonSym,0);
@test norm(A*X - B)/norm(B) < 1e-4
X, = solveLinearSystem(A,Bs,sNonSym,1);
@test norm(A'*X - Bs)/norm(Bs) < 1e-4

println("===  End Test Parallel Julia Solver ====");
