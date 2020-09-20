export getFacesStaggeredIndicesOfCellNoPressure,getFacesStaggeredIndicesOfCell,getCellCenteredIndicesOfCell,getNodalIndicesOfCell,getSubMeshOfCell

export getDirichletMassNodal


function getBoxIdxs(UpperLeftCorner::Array{Int64},BottomRightCorner::Array{Int64})
	# this is a type of an ndgrid function
	if length(UpperLeftCorner)==2
		i1 = vec(UpperLeftCorner[1]:BottomRightCorner[1]);
		i2 = vec(UpperLeftCorner[2]:BottomRightCorner[2]);
		I1 = i1*ones(Int64,length(i2))';
		I2 = ones(Int64,length(i1))*i2';
		return I1,I2
	else
		vx = UpperLeftCorner[1]:BottomRightCorner[1];
		vy = UpperLeftCorner[2]:BottomRightCorner[2];
		vz = UpperLeftCorner[3]:BottomRightCorner[3];
		m, n, o = length(vy), length(vx), length(vz)
		vx = reshape(vx, 1, n, 1)
		vy = reshape(vy, m, 1, 1)
		vz = reshape(vz, 1, 1, o)
		om = ones(Int64, m)
		on = ones(Int64, n)
		oo = ones(Int64, o)
		return (vx[om, :, oo], vy[:, on, oo], vz[om, on, :])
	end
end

##
# This function just extends the box with overlapp only if possible.
##

function getBoxWithOverlap(UpperLeftCorner::Array{Int64},BottomRightCorner::Array{Int64},nc::Array{Int64},overlap::Array{Int64})
	if length(nc)==2
		newUpperLeft = copy(UpperLeftCorner);
		newBottomRight = copy(BottomRightCorner);
		if UpperLeftCorner[1] > 1
			newUpperLeft[1]-=overlap[1];
		end
		if UpperLeftCorner[2] > 1
			newUpperLeft[2]-=overlap[2];
		end
		if BottomRightCorner[1] < nc[1]
			newBottomRight[1]+=overlap[1];
		end
		if BottomRightCorner[2] < nc[2]
			newBottomRight[2]+=overlap[2];
		end
	else
		newUpperLeft = copy(UpperLeftCorner);
		newBottomRight = copy(BottomRightCorner);
		for i=1:3
			if UpperLeftCorner[i] > 1
				newUpperLeft[i]-=overlap[i];
			end
		end
		for i=1:3
			if BottomRightCorner[i] < nc[i]
				newBottomRight[i]+=overlap[i];
			end
		end
	end
	return (newUpperLeft,newBottomRight);
end

function getCellCenteredIndicesOfCell(NumCells::Array{Int64,1},overlap::Array{Int64,1},i,nc)
	indOver = [];
	if length(nc)==2
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1];
		# CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc,overlap);
		I1,I2 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indOver = I1[:] + (I2[:]-1)*(nc[1])
	else
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1,(i[3]-1)*div(nc[3],NumCells[3]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1,div(nc[3],NumCells[3])-1];
		# CELLS X CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc,overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indOver = I1[:] + (I2[:]-1)*(nc[1]) + (I3[:]-1)*(nc[1]*nc[2]);
	end
	return indOver;
end


function getSubMeshOfCell(NumCells::Array{Int64,1},overlap::Array{Int64,1},i,Mesh::RegularMesh)
	indOver = [];
	nc = Mesh.n;
	if length(nc)==2
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1];
		# CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc,overlap);
	else
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1,(i[3]-1)*div(nc[3],NumCells[3]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1,div(nc[3],NumCells[3])-1];
		# CELLS X CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc,overlap);
	end
	subDomain = copy(Mesh.domain);
	subDomain[1:2:end] = Mesh.domain[1:2:end] .+ (UpperLeftCorner .- 1).*Mesh.h;
	subDomain[2:2:end] = Mesh.domain[2:2:end] .- (Mesh.n - BottomRightCorner).*Mesh.h;
	subMesh = getRegularMesh(subDomain,BottomRightCorner - UpperLeftCorner .+ 1);
	return subMesh
end


function getNodalIndicesOfCell(NumCells::Array{Int64,1},overlap::Array{Int64,1},i,nc)
	
	if length(nc)==2
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1]),div(nc[2],NumCells[2])];
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc .+ [1;1],overlap);
		I1,I2 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indNodal = I1[:] + (I2[:].-1)*(nc[1]+1)
	else
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1,(i[3]-1)*div(nc[3],NumCells[3]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1]),div(nc[2],NumCells[2]),div(nc[3],NumCells[3])];
		
		# NODES X NODES X NODES
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc + [1;1;1],overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indNodal = I1[:] + (I2[:].-1)*(nc[1]+1) + (I3[:].-1)*((nc[1]+1)*(nc[1]+1));
	end
	

	return indNodal
end


function getDirichletMassNodal(NumCells::Array{Int64},overlap::Array{Int64},i::Array{Int64},nc::Array{Int64})
	if length(nc)==2
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1]),div(nc[2],NumCells[2])];
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc .+ [1;1],overlap);
		size = BottomRightCorner-UpperLeftCorner.+1;
		mass = zeros(Float64,tuple(size...));
		if UpperLeftCorner[1]>1
			mass[1,:].=1.0;
		end
		if UpperLeftCorner[2]>1
			mass[:,1].=1.0;
		end
		if BottomRightCorner[1] < nc[1]+1
			mass[end,:].=1.0;
		end
		if BottomRightCorner[2] < nc[2]+1
			mass[:,end].=1.0;
		end
	else
		error("Not fully implemented")
		# originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1,(i[3]-1)*div(nc[3],NumCells[3]) + 1];
		# originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1]),div(nc[2],NumCells[2]),div(nc[3],NumCells[3])];
		# # NODES X NODES X NODES
		# (UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc + [1;1;1],overlap);
	end
	return mass;
end



function getFacesStaggeredIndicesOfCell(NumCells::Array{Int64},overlap::Array{Int64},i::Array{Int64},nc::Array{Int64})
	indOver = [];
	if length(nc)==2
		nf = [prod(nc + [1; 0]),prod(nc + [0; 1])];

		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1];
		# NODES X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [1;0],nc + [1;0],overlap);
		I1,I2 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indNC = I1[:] + (I2[:]-1)*(nc[1]+1)
	
		# CELLS X NODES
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [0;1],nc + [0;1],overlap);
		I1,I2 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCN = I1[:] + (I2[:]-1)*(nc[1])
	
		# CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc,overlap);
		I1,I2 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCC = I1[:] + (I2[:]-1)*(nc[1])
	
		indOver = [indNC; indCN + nf[1] ; indCC + nf[1]+nf[2]];
	else
		nf = [prod(nc + [1; 0 ; 0]),prod(nc + [0; 1 ; 0]),prod(nc + [0; 0 ; 1])];
		# overlap = [4,4,4];
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1,(i[3]-1)*div(nc[3],NumCells[3]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1,div(nc[3],NumCells[3])-1];
		
		# NODES X CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner + [1;0;0],nc + [1;0;0],overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indNCC = I1[:] + (I2[:]-1)*(nc[1]+1) + (I3[:]-1)*((nc[1]+1)*nc[2]);
	
		# CELLS X NODES X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [0;1;0],nc + [0;1;0],overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCNC = I1[:] + (I2[:]-1)*(nc[1]) + (I3[:]-1)*(nc[1]*(nc[2]+1));
		
		# CELLS X CELLS X NODES
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [0;0;1],nc + [0;0;1],overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCCN = I1[:] + (I2[:]-1)*(nc[1]) + (I3[:]-1)*(nc[1]*nc[2]);
		
		# CELLS X CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner,nc,overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCCC = I1[:] + (I2[:]-1)*(nc[1]) + (I3[:]-1)*(nc[1]*nc[2]);
		
		indOver = [indNCC; indCNC + nf[1] ; indCCN + nf[1]+nf[2] ; indCCC + nf[1]+nf[2]+nf[3]];
	end
	
	
	return indOver
end




function getFacesStaggeredIndicesOfCellNoPressure(NumCells::Array{Int64},overlap::Array{Int64},i::Array{Int64},nc::Array{Int64})
	indOver = [];
	if length(nc)==2
		nf = [prod(nc + [1; 0]),prod(nc + [0; 1])];

		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1];
		
		# NODES X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [1;0],nc + [1;0],overlap);
		I1,I2 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indNC = I1[:] .+ (I2[:].-1)*(nc[1]+1)
	
		# CELLS X NODES
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [0;1],nc + [0;1],overlap);
		I1,I2 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCN = I1[:] .+ (I2[:].-1)*(nc[1])
		
		indOver = [indNC; indCN .+ nf[1]];
		
	else
		nf = [prod(nc + [1; 0 ; 0]),prod(nc + [0; 1 ; 0]),prod(nc + [0; 0 ; 1])];
		originalUpperLeftCorner = [(i[1]-1)*div(nc[1],NumCells[1]) + 1,(i[2]-1)*div(nc[2],NumCells[2]) + 1,(i[3]-1)*div(nc[3],NumCells[3]) + 1];
		originalBottomRightCorner = originalUpperLeftCorner + [div(nc[1],NumCells[1])-1,div(nc[2],NumCells[2])-1,div(nc[3],NumCells[3])-1];
		
		# NODES X CELLS X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner + [1;0;0],nc + [1;0;0],overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indNCC = I1[:] + (I2[:]-1)*(nc[1]+1) + (I3[:].-1)*((nc[1]+1)*nc[2]);
	
		# CELLS X NODES X CELLS
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [0;1;0],nc + [0;1;0],overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCNC = I1[:] .+ (I2[:].-1)*(nc[1]) + (I3[:].-1)*(nc[1]*(nc[2]+1));
		
		# CELLS X CELLS X NODES
		(UpperLeftCorner,BottomRightCorner) = getBoxWithOverlap(originalUpperLeftCorner,originalBottomRightCorner+ [0;0;1],nc + [0;0;1],overlap);
		I1,I2,I3 = getBoxIdxs(UpperLeftCorner,BottomRightCorner);
		indCCN = I1[:] .+ (I2[:].-1)*(nc[1]) + (I3[:].-1)*(nc[1]*nc[2]);
		
		indOver = [indNCC; indCNC .+ nf[1] ; indCCN .+ (nf[1]+nf[2]) ];

	end
	return indOver 
end




