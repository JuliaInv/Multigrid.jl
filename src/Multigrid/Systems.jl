export getLinearOperatorsSystemsFaces,getInjectionOperatorsSystemsFaces,MGsetupSystems,getFacesStaggeredIndicesOfCell
## n is always in cells in this file

function speye(n)
	return sparse(1.0I,n,n);
end

function getInjectionOperatorsSystemsFaces(n::Array{Int64},withCellsBlock::Bool)
	if length(n) == 2
		R1, = getRestrictionFacesInjectionUj(n,1);
		R2, = getRestrictionFacesInjectionUj(n,2);
		
		if withCellsBlock
			R3, = getRestrictionCellCentered(n);
			Rinj = blockdiag(R1,R2,R3);
		else
			Rinj = blockdiag(R1,R2);
		end
	else
		R1, = getRestrictionFacesInjectionUj(n,1);
		R2, = getRestrictionFacesInjectionUj(n,2);
		R3, = getRestrictionFacesInjectionUj(n,3);
		if withCellsBlock
			R4, = getRestrictionCellCentered(n);
			Rinj = blockdiag(R1,R2,R3,R4);
		else
			Rinj = blockdiag(R1,R2,R3);
		end
	end
	return Rinj;
end

function getLinearOperatorsSystemsFaces(n::Array{Int64},withCellsBlock::Bool)
	if length(n) == 2
		(P1,nc) = getLinearInterpolationFacesUj(n,1);
		P2, = getLinearInterpolationFacesUj(n,2);
		if withCellsBlock
			P3, = getLinearInterpolationCellCentered(n);
			P  = blockdiag(P1,P2,P3);
		else
			P  = blockdiag(P1,P2);
		end
		R1, = getRestrictionFacesFullWeightUj(n,1);
		R2, = getRestrictionFacesFullWeightUj(n,2);
		# R1, = getRestrictionFacesInjectionUj(n,1);
		# R2, = getRestrictionFacesInjectionUj(n,2);
		
		if withCellsBlock
			R3, = getRestrictionCellCentered(n);
			R  = blockdiag(R1,R2,R3);
		else
			R  = blockdiag(R1,R2);
		end
	else
		(P1,nc) = getLinearInterpolationFacesUj(n,1);
		P2, = getLinearInterpolationFacesUj(n,2);
		P3, = getLinearInterpolationFacesUj(n,3);
		
		if withCellsBlock
			P4, = getLinearInterpolationCellCentered(n);
			P  = blockdiag(P1,P2,P3,P4);
		else
			P  = blockdiag(P1,P2,P3);
		end
		R1, = getRestrictionFacesFullWeightUj(n,1);
		R2, = getRestrictionFacesFullWeightUj(n,2);
		R3, = getRestrictionFacesFullWeightUj(n,3);
		if withCellsBlock
			R4, = getRestrictionCellCentered(n);
			R  = blockdiag(R1,R2,R3,R4);
		else
			R  = blockdiag(R1,R2,R3);
		end
	end
	return (P,R,nc);
end

function get1DRestrictionCells(n::Int64)
	## R is an 2X1 aggregation restriction.
	if n < 8
		return speye(n),n;
	end
	nc = div(n,2);
	if 2*nc != n
		error("Err: get1DRestrictionCells(): size should be a multiplication of 2");
	end
	#R = 1/2*spdiagm((ones(n-1),ones(n-1)),[0,1],n-1,n);
	I,J,V = SparseArrays.spdiagm_internal(0 => fill(.5,n-1), 1 => fill(.5,n-1))
	R = sparse(I, J, V, n-1, n);
	R = R[1:2:n,:];
	return R,nc;
end

function get1DNodeInjection(n_cells::Int64)
	## R is a node injection operator - C,F,C,F,...,C
	n = n_cells;
	if n < 8
		return speye(n+1),n;
	end
	nc = div(n,2);
	if 2*nc != n
		error("Err: get1DNodeInjection(): size should be a multiplication of 2");
	end
	R = speye(n+1);
	R = R[1:2:n+1,:];
	return R,nc;
end

function get1DNodeFullWeightRestriction(n_cells::Int64)
	## R is a full weighting X2 restriction: 0.25 , 0.5 , 0.25 operates on nodes.
	## B.C: Here we need to choose what happens at boundaries - we choose injection
	n = n_cells;
	if n < 8
		return speye(n+1),n;
	end
	nc = div(n,2);
	if 2*nc != n
		error("Err: get1DNodeFullWeightRestriction(): size should be a multiplication of 2");
	end
	#Qtr = 0.25*ones(n);
	#R = spdiagm((Qtr,0.5*ones(n+1),Qtr),[-1,0,1],n+1,n+1);
	R = spdiagm(-1=>fill(.25,n), 0=>fill(.5,n+1) , 1=>fill(.25,n));
	
    R = sparse(R[:,1:2:end]');
	R[1,1:2] = [0.75 0.25];
	R[end,end-1:end] = [0.25 0.75];
	return R,nc;
end


function get1DProlongationCellCentered(ncells_fine::Int64)
	## P takes a two cells [C,C] into [F,F,F,F] 
	n = ncells_fine;
	if n < 8
		return speye(n),n;
	end
	nc = div(n,2);
	if 2*nc != n
		error("Err: get1DProlongationCellCentered(): size should be a multiplication of 2");
	end
	#P  = spdiagm(((1/4)*ones(n-2),(3/4)*ones(n-1),(3/4)*ones(n),(1/4)*ones(n-1)),[-2,-1,0,1],n,n);
	P = spdiagm(-2=> fill(.25,n-2), -1=>fill(.75,n-1), 0=>fill(.75,n-1) , 1=>fill(.25,n-1));
	P = P[:,1:2:end];
	P[1,1:2] = [5/4,-1/4];
	P[end,end-1:end] = [-1/4,5/4];
	return P,nc;
end

function get1DProlongationNodes(ncells_fine::Int64)
	## P takes a two cells [C,C] into [F,F,F,F] 
	n = ncells_fine;
	if n < 8
		return speye(n+1),n;
	end
	nc = div(n,2);
	if 2*nc != n
		error("Err: get1DProlongationNodes(): size should be a multiplication of 2");
	end
	Half = 0.5*ones(n);
	#P = spdiagm((Half,ones(n+1),Half),[-1,0,1],n+1,n+1);
	P = spdiagm(-1=>Half, 0=>ones(n+1), 1=>Half);
    P = P[:,1:2:end];
	return P,nc;
end


function getRestrictionCellCentered(n::Array{Int64})	
	if length(n) == 3
		(R1,nc1) = get1DRestrictionCells(n[1]);
		(R2,nc2) = get1DRestrictionCells(n[2]);
		(R3,nc3) = get1DRestrictionCells(n[3]);
		R = kron(R3,kron(R2,R1));
		nc = [nc1;nc2;nc3];
	elseif length(n) == 2
		(R1,nc1) = get1DRestrictionCells(n[1]);
		(R2,nc2) = get1DRestrictionCells(n[2]);
		R = kron(R2,R1);
		nc = [nc1;nc2];
	else
		error("getRestrictionCells() : Dimension not supported!");
	end	
	return R,nc
end


function getRestrictionFacesInjectionUj(n::Array{Int64},j::Int64)
    # n here is number of cells.
	R = Array{SparseMatrixCSC}(length(n));
	nc = zeros(Int64,length(n));
	for kk = 1:length(n)
		if kk == j
			(R[kk],nc[kk]) = get1DNodeInjection(n[kk]);
		else
			(R[kk],nc[kk]) = get1DRestrictionCells(n[kk]);
		end
	end
	if length(n) == 3
		R = kron(R[3],kron(R[2],R[1]));
	elseif length(n) == 2
		R = kron(R[2],R[1]);
	else
		error("getRestrictionFacesInjectionUj() : Dimension not supported!");
	end	
	return R,nc;
end

function getRestrictionFacesFullWeightUj(n::Array{Int64},j::Int64)
    # n here is number of cells.
	R = Array{SparseMatrixCSC}(undef,length(n));
	nc = zeros(Int64,length(n));
	for kk = 1:length(n)
		if kk == j
			(R[kk],nc[kk]) = get1DNodeFullWeightRestriction(n[kk]);
		else
			(R[kk],nc[kk]) = get1DRestrictionCells(n[kk]);
		end
	end
	if length(n) == 3
		R = kron(R[3],kron(R[2],R[1]));
	elseif length(n) == 2
		R = kron(R[2],R[1]);
	else
		error("getRestrictionFacesFullWeightUj() : Dimension not supported!");
	end	
	return R,nc;
end

function getLinearInterpolationFacesUj(n::Array{Int64},j::Int64)
	P = Array{SparseMatrixCSC}(undef,length(n));
	nc = zeros(Int64,length(n));
	for kk = 1:length(n)
		if kk == j
			(P[kk],nc[kk]) = get1DProlongationNodes(n[kk]);
		else
			(P[kk],nc[kk]) = get1DProlongationCellCentered(n[kk]);
		end
	end
	if length(n) == 3
		P = kron(P[3],kron(P[2],P[1]));
	elseif length(n) == 2
		P = kron(P[2],P[1]);
	else
		error("getLinearInterpolationFacesUj() : Dimension not supported!");
	end	
	return P,nc;
end

function getLinearInterpolationCellCentered(n::Array{Int64})
	nc = zeros(Int64,length(n));
	if length(n) == 3
		(P1,nc1) = get1DProlongationCellCentered(n[1]);
		(P2,nc2) = get1DProlongationCellCentered(n[2]);
		(P3,nc3) = get1DProlongationCellCentered(n[3]);
		P = kron(P3,kron(P2,P1));
		nc = [nc1;nc2;nc3];
	elseif length(n) == 2
		(P1,nc1) = get1DProlongationCellCentered(n[1]);
		(P2,nc2) = get1DProlongationCellCentered(n[2]);
		P = kron(P2,P1);
		nc = [nc1;nc2];
	else
		error("getLinearInterpolationCellCentered() : Dimension not supported!");
	end	
	return P,nc;
end


