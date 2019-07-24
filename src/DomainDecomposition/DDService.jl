export getIndicesOfCellsArray;
function getIndicesOfCellsArray(param::DomainDecompositionParam)
	Mesh 				= param.Mesh;
	ncells				= Mesh.n;
	overlap 			= param.overlap;
	numDomains 			= param.numDomains
	getIndicesOfCell	= param.getIndicesOfCell;
	Idxs = getIndicesOfCell(numDomains,overlap,div.(ncells,2)+1,ncells);
	lenIdxs = length(Idxs)
	ArrIdxs = zeros(UInt32,lenIdxs,prod(numDomains));
	for ic = 1:prod(numDomains);
		icloc = cs2loc(ic,numDomains);
		IIp = getIndicesOfCell(numDomains,overlap,icloc,ncells);
		ArrIdxs[1:length(IIp),ic] = IIp;
	end
	return ArrIdxs;
end

export loc2cs3D
function loc2cs3D(loc::Array{Int64,1},n::Array{Int64,1})
@inbounds cs = loc[1] + (loc[2]-1)*n[1] + (loc[3]-1)*n[1]*n[2];
return cs;
end

export loc2cs
function loc2cs(loc::Array{Int64,1},n::Array{Int64,1})
if length(n)==2
	@inbounds cs = loc[1] + (loc[2]-1)*n[1];
else
	@inbounds cs = loc[1] + (loc[2]-1)*n[1] + (loc[3]-1)*n[1]*n[2];
end
return cs;
end

export cs2loc
function cs2loc(cs_loc::Int64,n::Array{Int64,1})
	if length(n)==3
		@inbounds loc1 = mod(cs_loc-1,n[1])+1;
		@inbounds loc2 = div(mod(cs_loc-1,n[1]*n[2]),n[1]) + 1;
		@inbounds loc3 = div(cs_loc-1,n[1]*n[2])+1;
		return [loc1;loc2;loc3];
	else
		@inbounds loc1 = mod(cs_loc-1,n[1]) + 1;
		@inbounds loc2 = div(cs_loc-1,n[1])+1;
		return [loc1;loc2];
	end
end
