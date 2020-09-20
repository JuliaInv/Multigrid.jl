function getSubParams(Hparam::RegularMesh, M::RegularMesh,i::Array{Int64},NumCells::Array{Int64},overlap::Array{Int64})
	subMesh   = getSubMeshOfCell(NumCells,overlap,i,Hparam);
	# IIp       = getCellCenteredIndicesOfCell(NumCells,overlap,i,M.n);
	# Hparam    = ElasticHelmholtzParam(subMesh,w,Hparam.lambda[IIp],Hparam.rho[IIp],Hparam.mu[IIp],Hparam.gamma[IIp]+subgamma[:],false);
	# return Hparam;
	return subMesh;
end

function getLap(M::RegularMesh)
	# print("getLap: ",M.n);
	G = getNodalGradientMatrix(M);
	Ar = G'*G;
	Ar = Ar + 1e-5*opnorm(Ar,1)*sparse(1.0I,size(Ar,2),size(Ar,2));
	# println(", done!")
	return Ar;
end

function getDirichletMassNodalMesh(DDparam::DomainDecompositionParam,i::Array{Int64})
	d = getDirichletMassNodal(DDparam.numDomains,DDparam.overlap,i,DDparam.Mesh.n);
	d.*=(0.1*4.0)/prod(DDparam.Mesh.h);
	return d;
end