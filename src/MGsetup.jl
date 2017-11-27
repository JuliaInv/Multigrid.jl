export MGsetup,transposeHierarchy,adjustMemoryForNumRHS,getSPAIprec,getFWInterp,replaceMatrixInHierarchy
export restrictCellCenteredVariables,restrictNodalVariables2;
single = false;
function MGsetup(ATf::Union{SparseMatrixCSC,multilevelOperatorConstructor},Mesh::RegularMesh,param::MGparam,rhsType::DataType = Float64,nrhs::Int64 = 1,verbose::Bool=false)
Ps      	= Array{SparseMatrixCSC}(param.levels-1);
Rs      	= Array{SparseMatrixCSC}(param.levels-1);
As 			= Array{SparseMatrixCSC}(param.levels);
Meshes  	= Array{RegularMesh}(param.levels); 
relaxPrecs 	= Array{Any}(param.levels);

PDEparam = 0;
if isa(ATf,SparseMatrixCSC)==true
	As[1] = ATf;
else
	As[1] = ATf.getOperator(Mesh,ATf.param)';
	PDEparam = ATf.param;
end
Meshes[1] = Mesh;
n = Mesh.n;
AT = As[1];
Cop = nnz(AT);
for l = 1:(param.levels-1)
	if verbose
		tic()
	end
    AT = As[l];
    
	PT = 0;
	RT = 0;
	withCellsBlock = false;
	if param.transferOperatorType=="SystemsFacesMixedLinear"
		withCellsBlock = true;
	end
	
	if param.transferOperatorType=="FullWeighting"
		geometric = !isa(ATf,SparseMatrixCSC);
		(P,nc) = getFWInterp(n+1,geometric);
		if isa(ATf,SparseMatrixCSC)
			RT = P;
			PT = P';
		else
			RT = (0.5^Meshes[l].dim)*P;
			PT = P';
		end
		nc = nc-1;
		P = 0;
	elseif param.transferOperatorType=="SystemsFacesLinear" || param.transferOperatorType=="SystemsFacesMixedLinear"
		(P,R,nc) = getLinearOperatorsSystemsFaces(n,withCellsBlock);
		PT = P';
		RT = R';
		P = 0;
		R = 0;
	end
	
	if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES"
		d = param.relaxParam./diag(AT);
		d = spdiagm(d);
		if single
			relaxPrecs[l] = convert(SparseMatrixCSC{Complex64,Int32},d);# here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels
		else
			relaxPrecs[l] = d;
		end
		
	elseif param.relaxType=="SPAI"
		relaxPrecs[l] = spdiagm(param.relaxParam*getSPAIprec(AT)); # here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels
	elseif param.relaxType=="VankaFaces"
		relaxPrecs[l] = getVankaFacesPreconditioner(AT,Meshes[l],param.relaxParam,withCellsBlock);
	else
		error("Unknown relaxation type !!!!");
	end
	
	if (size(PT,1)==size(PT,2))
		if verbose; println(string("Stopped Coarsening at level ",l)); end
		param.levels = l;
		As = As[1:l];
		Ps = Ps[1:l-1];
		Rs = Rs[1:l-1];
		Meshes = Meshes[1:l];
		relaxPrecs = relaxPrecs[1:l];
		break;
	else
		Ps[l] = PT;
		Rs[l] = RT;
		Meshes[l+1] = getRegularMesh(Meshes[l].domain,nc);
		if isa(ATf,SparseMatrixCSC)
			# if param.transferOperatorType=="SystemsFacesLinear"
				# Rinj = getInjectionOperatorsSystemsFaces(n,withCellsBlock)';
				# Act = Ps[l]*AT*Rinj;
			# else
				Act = Ps[l]*AT*Rs[l];
			# end
		else
			PDEparam = ATf.restrictParams(Meshes[l],Meshes[l+1],PDEparam,l);
			Act = ATf.getOperator(Meshes[l+1],PDEparam)';
		end		
		As[l+1] = Act;
		Cop = Cop + nnz(Act);
		if verbose; println("MG setup: ",n," cells took:",toq()); end;
		n = nc;
	end
end
if verbose 
	tic()
	println("MG setup: Operator complexity = ",Cop/nnz(As[1]));
end

param.As = As;
param.Meshes = Meshes;

defineCoarsestAinv(param,As[end]);

if verbose 
	println("MG setup coarsest ",param.coarseSolveType,": ",n," cells , done LU in ",toq());
end
param.Ps = Ps;
param.Rs = Rs;
param.relaxPrecs = relaxPrecs;
param = adjustMemoryForNumRHS(param,rhsType,nrhs,verbose);
param.doTranspose = 0;
return;
end

function adjustMemoryForNumRHS(param::MGparam,rhsType::DataType = Float64,nrhs::Int64 = 1,verbose::Bool=false)
if length(param.As)==0
	error("The Hierarchy is empty - run a setup first.")
end
good = true;
if length(param.memCycle) > 0
	if size(param.memCycle[1].x,2) != nrhs 
		good = false
	elseif param.relaxType == "Jac-GMRES"
		if length(param.memRelax)==0
			good = false;
		end
	elseif param.cycleType == 'K'
		if length(param.memRelax)==0
			good = false;
		end
	end
else
	good = false;
end
if good
	return param;
end

if param.cycleType == 'K'
	memKcycle = Array{FGMRESmem}(max(param.levels-2,0)); 
else
	memKcycle = Array{FGMRESmem}(0);
end

if param.relaxType == "Jac-GMRES"
	memRelax = Array{FGMRESmem}(param.levels-1); 
else
	memRelax = Array{FGMRESmem}(0);
end

memCycle = Array{CYCLEmem}(param.levels);

param.memRelax  = memRelax;
param.memKcycle = memKcycle;
param.memCycle  = memCycle;

N = size(param.As[1],2); 

for l = 1:(param.levels-1)
	N = size(param.As[l],2);
	needZ = param.relaxType=="Jac-GMRES";
	memCycle[l] = getCYCLEmem(N,nrhs,rhsType,true);
	if param.relaxType=="Jac-GMRES"
		maxRelax = max(param.relaxPre(l),param.relaxPost(l));
		if nrhs == 1
			memRelax[l] = getFGMRESmem(N,true,rhsType,maxRelax,1);
		else
			# memRelax[l] = getBlockFGMRESmem(N,nrhs,false,rhsType,maxRelax);
			# memRelax[l] = getFGMRESmem(N,false,rhsType,maxRelax,nrhs);
			memRelax[l] = getFGMRESmem(N,true,rhsType,maxRelax,1);
		end
	end
	
	if l > 1 && param.cycleType=='K'
		if nrhs == 1
			memKcycle[l-1] = getFGMRESmem(N,true,rhsType,2,1);
		else
			# memKcycle[l-1] = getBlockFGMRESmem(N,nrhs,true,rhsType,2);
			memKcycle[l-1] = getFGMRESmem(N,true,rhsType,2,nrhs);
		end
		
	end
end
memCycle[end] = getCYCLEmem(size(param.As[end],2),nrhs,rhsType,false); # no need for residual on the coarsest level...
memCycle[end].b = memCycle[end].r;
param.memRelax = memRelax;
param.memKcycle = memKcycle;
param.memCycle = memCycle;
return param;
end


function replaceMatrixInHierarchy(param,AT::SparseMatrixCSC,verbose::Bool=false)

param.As[1] = AT;
Cop = nnz(AT);
for l = 1:(param.levels-1)
	if verbose
		tic()
	end
    AT = param.As[l];
	
	if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES"
		d = param.relaxParam./diag(AT);
		d = spdiagm(d);
		param.relaxPrecs[l] = d;
		if single
			param.relaxPrecs[l] = convert(SparseMatrixCSC{Complex64,Int32},d);# here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels
		end
		
	elseif param.relaxType=="SPAI"
		d = spdiagm(d);
	elseif param.relaxType=="VankaFaces"
		param.relaxPrecs[l] = getVankaFacesPreconditioner(AT,Meshes[l],param.relaxParam,withCellsBlock);
	else
		error("Unknown relaxation type !!!!");
	end
	Act = param.Ps[l]*AT*param.Rs[l];
	param.As[l+1] = Act;
	Cop = Cop + nnz(Act);
	if verbose; println("MG setup: took:",toq()); end;
end
if verbose 
	tic()
	println("MG setup: Operator complexity = ",Cop/nnz(param.As[1]));
end

defineCoarsestAinv(param,param.As[end]);

if verbose 
	println("MG setup coarsest ",param.coarseSolveType,":  done LU in ",toq());
end
param.doTranspose = 0;
return;
end



function transposeHierarchy(param::MGparam,verbose::Bool=false)
# This is a shortened MG setup. It assumes that a setup was previously applied 
# to the matrix in param.As[1] (which is held transposed). It keeps the same Ps and memory allocations.
# Now all hierarchy is transposed. doTranspose is set to 1. MUMPS do not replace the factorization cause it knows to solve transpose.

param.As[1] = param.As[1]';
param.doTranspose = mod(param.doTranspose+1,2);

for l = 1:(param.levels-1)
	if verbose
		println("MG transpose level: ",l);
		tic()
	end

    if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES" || param.relaxType=="SPAI"
		param.relaxPrecs[l] = conj(param.relaxPrecs[l]);
	else
		error("Not supported");
	end
	R = param.Ps[l]';
	param.Ps[l] = param.Rs[l]';
	param.Rs[l] = R;
	
	param.As[l+1]  = param.As[l+1]';
end
if verbose 
	tic()
end
if param.coarseSolveType == "MUMPS"
	# when applying mumps, no need to transpose the factorization
elseif param.coarseSolveType == "BiCGSTAB" || param.coarseSolveType == "GMRES"
	param.LU = conj(param.LU);
else
	destroyCoarsestLU(param);
	param.LU = lufact(param.As[end]');
end
if verbose 
	println("MG transpose hierarchy: done LU");
	toc();
end
return;
end



export defineCoarsestAinv;
function defineCoarsestAinv(param::MGparam,AT::SparseMatrixCSC)
if param.coarseSolveType == "MUMPS"
	param.LU = factorMUMPS(AT',0,0);
elseif param.coarseSolveType == "BiCGSTAB" || param.coarseSolveType == "GMRES"
	d = spdiagm(param.relaxParam./diag(AT));
	param.LU = d;
	if single
		param.LU = convert(SparseMatrixCSC{Complex64,Int32},d);# here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels	
	end
	
# elseif param.coarseSolveType[1:7] == "DDNodal"
	# DDNumCells = parseDD(param.coarseSolveType,param.Meshes[end].dim);
	# param.LU,nnzDD = setupDD(AT,param.Meshes[end],getNodalIndicesOfCell,DDNumCells,1e-5);
# elseif param.coarseSolveType[1:7] == "DDFaces"
	# DDNumCells = parseDD(param.coarseSolveType,param.Meshes[end].dim);
	# param.LU,nnzDD = setupDD(AT,param.Meshes[end],getFacesStaggeredIndicesOfCell, DDNumCells,1e-5);
	# # println("Cop coarsest (DomainDecomp): ",nnzDD./nnz(param.As[1]));
else
	param.LU = lufact(AT');
	# nnzLU = nnz(param.LU[:L]) + nnz(param.LU[:U]);
	# println("Cop coarsest (Julia): ",nnzLU./nnz(param.As[1]));
end
end


# function parseDD(DDstring::String,dim::Int64)
# if dim==2
	# DDNumCells = [parse(Int64,DDstring[end-1]),parse(Int64,DDstring[end])];
# else
	# DDNumCells = [parse(Int64,DDstring[end-2]),parse(Int64,DDstring[end-1]),parse(Int64,DDstring[end])];
# end
# return DDNumCells;
# end





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
	P = spdiagm((halfVec,ones(n_nodes),halfVec),[-1,0,1],n_nodes,n_nodes);
    if oddDim == 1
        P = P[:,1:2:end];
    else
		if geometric
			P = speye(n_nodes);
			warn("getFWInterp: in geometric mode we stop coarsening because num cells does not divide by two");
		else 
			P = P[:,[1:2:end;end]];
			P[end-1:end,end-1:end] = speye(2);
#         	P = P[:,1:2:end];
#         	P[end,end-1:end] = [-0.5,1.5];
		end
    end
else
    P = speye(n_nodes);
end
nc = size(P,2);
if single
	P = convert(SparseMatrixCSC{Float32,Int32},P);
end
return P,nc
end


function getSPAIprec(A::SparseMatrixCSC)
s = vec(sum(real(A).^2,2) + sum(imag(A).^2,2));
Q = conj(diag(A))./s;
end


########################### GEOMETRIC MULTIGRID STUFF ###############################

function restrictCellCenteredVariables(rho::ArrayTypes,n::Array{Int64})
R,nc = getRestrictionCellCentered(n);
rho_c = R*rho[:];
return rho_c;
## TODO: make this more efficient...
end

export restrictNodalVariables
function restrictNodalVariables(rho::ArrayTypes,n_nodes::Array{Int64})
P,nc = getFWInterp(n_nodes,true);
rho_c = zeros(eltype(rho),size(P,2));
rho_c[:] = (0.5^length(n_nodes))*(P'*rho[:]);
return rho_c;
## TODO: make this more efficient...
end

function restrictNodalVariables2(rho::ArrayTypes,n_nodes::Array{Int64})
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




