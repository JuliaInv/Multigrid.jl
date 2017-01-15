using jInv.Mesh
using DivSigGrad
using Multigrid

println("************************************************* Example 2D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0];
n = [100,100];
Mr = getRegularMesh(domain,n)
m  = exp(2*randn(prod(n)));
Ar = getDivSigGradMatrix(vec(m),Mr);
# Ar = Ar + 1e-4*norm(Ar,1)*speye(size(Ar,2));

levels      = 4;
numCores 	= 8; 
maxIter     = 20;
relativeTol = 1e-6;
relaxType   = "SPAI";
relaxParam  = 1.0;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='V';
coarseSolveType = "MUMPS";

MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);

N = size(Ar,2);

b = Ar*rand(N,20);
x = zeros(N,20);
SA_AMGsetup(Ar,MG,Float64,true,1,true);


# println("****************************** Stand-alone AMG single rhs: ******************************")
# solveMG(MG,vec(b[:,1]),vec(x[:,1]),true);

# x[:] = 0.0;
# println("****************************** Stand-alone AMG multiple rhs: ******************************")
# solveMG(MG,b,x,true);


# x[:] = 0.0;
# println("****************************** CG preconditioned with AMG single rhs: ******************************")
# solveCG_MG(Ar,MG,vec(b[:,1]),vec(x[:,1]),true)

# println("****************************** CG preconditioned with AMG multiple rhs: ******************************")
# solveCG_MG(Ar,MG,b,x,true)



println("****************************** GMRES preconditioned with AMG: single rhs: ******************************")
solveGMRES_MG(Ar,MG,vec(b[:,1]),vec(x[:,1]),true,10)

println("****************************** GMRES preconditioned with AMG: multiple rhs: ******************************")
solveGMRES_MG(Ar,MG,b,x,true,10)



println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
error("Eran")
Ar = 0;
b = 0;
x = 0;
Mr = 0;
println("************************************************* Example 3D ******************************************************");

domain = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
n = [100,100,100];
Mr = getRegularMesh(domain,n)
m  = exp(randn(prod(n)));
Ar = getDivSigGradMatrix(vec(m),Mr);
Ar = Ar + 1e-6*norm(Ar,1)*speye(size(Ar,2));

MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,coarseSolveType);

N = size(Ar,2);
b = Ar*rand(N,3);
x = zeros(N,3);

SA_AMGsetup(Ar,MG,Float64,true,size(b,2),true);
println("Stand-alone AMG:")
solveMG(MG,b,x,true);

println("CG preconditioned with AMG:")
x[:] = 0.0;
solveCG_MG(Ar,MG,b,x,true)
