using jInv.Mesh;
using Multigrid

# function getNodalLaplacianMatrixDirichlet(Minv::RegularMesh)
# newDomain = copy(Minv.domain);
# newDomain[1] -= Minv.h[1];
# newDomain[2] += Minv.h[1];
# newDomain[3] -= Minv.h[2];
# newDomain[4] += Minv.h[2];
# MinvPadded = getRegularMesh(newDomain,Minv.n+2);
# mask = zeros(Bool,tuple(MinvPadded.n+1...));
# mask[2:end-1,2:end-1] = true;
# A = getNodalLaplacianMatrix(MinvPadded);
# A = A[mask[:],mask[:]];
# return A;
# end

# A = getNodalLaplacianMatrixDirichlet(getRegularMesh([0.0,1.0,0.0,1.0],[4,4]))


m = ones(129,129);

Minv = getRegularMesh([0.0,1.0,0.0,1.0],collect(size(m))-1);

levels      = 4;
numCores 	= 2; 
maxIter     = 5;
relativeTol = 1e-2;
relaxType   = "Jac";
relaxParam  = 0.8;
relaxPre 	= 2;
relaxPost   = 2;
cycleType   ='V';
coarseSolveType = "NoMUMPS";

MG = getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,
				relaxPre,relaxPost,cycleType,coarseSolveType,0.5,0.0)

## TEST DIRICHLET
# L = getNodalLaplacianMatrixDirichlet(Minv);
# s = zeros(tuple(Minv.n+1...));
# s[2:end-1,2:end-1] = rand(tuple(Minv.n-1...));
# s = s[:];
# MGsetup(getNodalLaplacianMatrixDirichlet,Minv,MG,Float64,1,true)
# MGsetup(L,Minv,MG,Float64,1,true)

## TEST NEUMANN
L = getNodalLaplacianMatrix(Minv);
s = rand(prod(Minv.n+1));
b = L*s;
x0 = 0.0*s;

MGsetup(getMultilevelOperatorConstructor([],getNodalLaplacianMatrix,[]),Minv,MG,Float64,1,true);
solveMG(MG,b,x0,true);

xn = getCellCenteredGrid(Minv);
sig = 3*xn[:,1].*(1.0-xn[:,1]) + 2*xn[:,2].*(1.0-xn[:,2]);
sig = sig[:];
# sig[:]=1.0;


## The following lines also appear in a different version in jInv.Mesh. 
## The difference is the averaging: in jInv.Mesh sigma is averaged with 0.0 outside the domain (there's a difference in how av is defined).
## geometric MG does not work with the jInv version. only with this one.

function av(n)
# A = av(n), 1D average operator
	av = spdiagm((fill(.5,n),fill(.5,n)),(-1,0),n+1,n);
	av[1,1] = 1.0;
	av[end,end] = 1.0;
	return av; 
end
function getEdgeAverageMatrix(Mesh::AbstractTensorMesh)
	if Mesh.dim==3
		A1 = kron(av(Mesh.n[3]),kron(av(Mesh.n[2]),speye(Mesh.n[1]))) 
		A2 = kron(av(Mesh.n[3]),kron(speye(Mesh.n[2]),av(Mesh.n[1]))) 
		A3 = kron(speye(Mesh.n[3]),kron(av(Mesh.n[2]),av(Mesh.n[1])))
		Mesh.Ae = [A1; A2; A3]
	elseif Mesh.dim==2
		A1 = kron(av(Mesh.n[2]),speye(Mesh.n[1]))
		A2 = kron(speye(Mesh.n[2]),av(Mesh.n[1]))
		Ae = [A1; A2]
	else
		error("getEdgeAverageMatrix not implemented fot $(Mesh.dim)D Meshes")
	end
	return Ae
end
function getDivSigGrad(M::AbstractTensorMesh,sig::Vector{Float64})
 	G       = getNodalGradientMatrix(M)
	Ae      = getEdgeAverageMatrix(M);
    A       = (G'*spdiagm(Ae*(vec(sig))))*G
	return A
end

Ar = getDivSigGrad(Minv,sig);
b = Ar*s;
MGsetup(Ar,Minv,MG,Float64,1,true);
x0 = 0.0*s;
solveMG(MG,b,x0,true);

# getDivSigGrad(mesh::RegularMesh,param::Array{Float64,1}) = getDivSigGradMatrix(param,mesh);
restrictSigma(mesh_fine,mesh_coarse,param_fine,level) = restrictCellCenteredVariables(param_fine,mesh_fine.n);
MGsetup(getMultilevelOperatorConstructor(sig,getDivSigGrad,restrictSigma),Minv,MG,Float64,1,true);
x0 = 0.0*s;
solveMG(MG,b,x0,true);
println("DONE!")