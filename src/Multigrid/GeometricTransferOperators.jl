export restrictCellCenteredVariables,restrictNodalVariables2,getFWInterp;


# Bilinear "Full Weighting" prolongation
function getFWInterp(n_nodes::Array{Int64,1},geometric::Bool=false)
# n here is the number of NODES!!!
(P1,nc1) = get1DFWInterp(n_nodes[1],geometric);
(P2,nc2) = get1DFWInterp(n_nodes[2],geometric);
if length(n_nodes)==3
	(P3,nc3) = get1DFWInterp(n_nodes[3],geometric);
end
if length(n_nodes)==2
	P = kron(P2,P1);
	nc = [nc1,nc2];
else
	P = kron(P3,kron(P2,P1));
	nc = [nc1,nc2,nc3];
end
return P,nc
end

function get1DFWInterp(n_nodes::Int64,geometric)
# n here is the number of NODES!!!
oddDim = mod(n_nodes,2);
if n_nodes > 2
	halfVec = 0.5*ones(n_nodes-1);
	P = spdiagm(-1=>halfVec,0=>ones(n_nodes),1=>halfVec); #used to be P = spdiagm((halfVec,ones(n_nodes),halfVec),[-1,0,1],n_nodes,n_nodes);
    if oddDim == 1
        P = P[:,1:2:end];
    else
		if geometric
			P = sparse(1.0I,n_nodes,n_nodes);
			println("Warning: getFWInterp(): in geometric mode we stop coarsening because num cells does not divide by two");
		else 
			P = P[:,[1:2:end;end]];
			P[end-1:end,end-1:end] = speye(2);
#         	P = P[:,1:2:end];
#         	P[end,end-1:end] = [-0.5,1.5];
		end
    end
else
    P = sparse(1.0I,n_nodes,n_nodes);
end
nc = size(P,2);
return P,nc
end



########################### GEOMETRIC MULTIGRID STUFF ###############################

function restrictCellCenteredVariables(rho::Array,n::Array{Int64})
R,nc = getRestrictionCellCentered(n);
rho_c = R*rho[:];
return rho_c;
## TODO: make this more efficient...
end

export restrictNodalVariables
function restrictNodalVariables(rho::Array,n_nodes::Array{Int64})
P,nc = getFWInterp(n_nodes,true);
rho_c = zeros(eltype(rho),size(P,2));
rho_c[:] = (0.5^length(n_nodes))*(P'*rho[:]);
return rho_c;
## TODO: make this more efficient...
end

function restrictNodalVariables2(rho::Array,n_nodes::Array{Int64})
P,nc = getFWInterp(n_nodes,true);
R1,nc1 = get1DNodeFullWeightRestriction(n_nodes[1]-1);
R2,nc2 = get1DNodeFullWeightRestriction(n_nodes[2]-1);
R = kron(R2,R1);
# rho_c = zeros(eltype(rho),size(P,2));
# println(size(R))
# println(size(rho[:]))

rho_c = R*rho[:];
return rho_c;
## TODO: make this more efficient...
end




