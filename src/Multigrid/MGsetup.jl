export MGsetup,transposeHierarchy,adjustMemoryForNumRHS,getSPAIprec,getFWInterp,replaceMatrixInHierarchy
export restrictCellCenteredVariables,restrictNodalVariables2;


const spIndType = Int64
export spIndType;

function MGsetup(ATf::Union{SparseMatrixCSC,multilevelOperatorConstructor},Mesh::RegularMesh,param::MGparam,rhsType::DataType = Float64,nrhs::Int64 = 1,verbose::Bool=false)
Ps      	= Array{SparseMatrixCSC}(undef,param.levels-1);
Rs      	= Array{SparseMatrixCSC}(undef,param.levels-1);
As 			= Array{SparseMatrixCSC}(undef,param.levels);
Meshes  	= Array{RegularMesh}(undef,param.levels); 
relaxPrecs 	= Array{Any}(undef,param.levels);

MGType = getMGType(param,rhsType);


PDEparam = 0;
if isa(ATf,SparseMatrixCSC)==true
	As[1] = ATf;
else
	As[1] = sparse(ATf.getOperator(Mesh,ATf.param)');
	PDEparam = ATf.param;
end
Meshes[1] = Mesh;
n = Mesh.n;
AT = As[1];
Cop = nnz(AT);
T_time = 0;

for l = 1:(param.levels-1)
	if verbose
		T_time = time_ns();
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
		(P,nc) = getFWInterp(n.+1,geometric);
		nc = nc.-1;
		if isa(ATf,SparseMatrixCSC)
			RT = P;
			PT = P';
		else
			RT = (0.5^Meshes[l].dim)*P;
			PT = sparse(P');
		end
		P = 0;
	elseif param.transferOperatorType=="SystemsFacesLinear" || param.transferOperatorType=="SystemsFacesMixedLinear"
		(P,R,nc) = getLinearOperatorsSystemsFaces(n,withCellsBlock);
		PT = sparse(P');
		RT = sparse(R');
		P = 0;
		R = 0;
	end
	
	if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES"
		d = param.relaxParam./diag(AT);
		d = sparse(Diagonal(d));
		if param.singlePrecision
			relaxPrecs[l] = convert(SparseMatrixCSC{MGType,spIndType},d);# here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels
		else
			relaxPrecs[l] = d;
		end
		
	elseif param.relaxType=="SPAI"
		relaxPrecs[l] = sparse(Diagonal(param.relaxParam*getSPAIprec(AT))); # here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels
	elseif param.relaxType=="VankaFaces"
		relaxPrecs[l] = getVankaFacesPreconditioner(AT,Meshes[l],param.relaxParam[l],withCellsBlock);
	else
		error("Unknown relaxation type !!!!");
	end
	if param.singlePrecision
		PT = convert(SparseMatrixCSC{real(MGType),spIndType},PT);
		RT = convert(SparseMatrixCSC{real(MGType),spIndType},RT);
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
			Act = sparse(ATf.getOperator(Meshes[l+1],PDEparam)');
		end
		As[l+1] = Act;
		Cop = Cop + nnz(Act);
		if verbose; println("MG setup: ",n," cells took:",(time_ns()-T_time)/1e+9); end;
		n = nc;
	end
end

if verbose 
	T_time = time_ns();
	println("MG setup: Operator complexity = ",Cop/nnz(As[1]));
end

param.As = As;
param.Meshes = Meshes;

defineCoarsestAinv(param,As[end]);

if verbose 
	println("MG setup coarsest ",param.coarseSolveType,": ",n," cells , done LU in ",(time_ns()-T_time)/1e+9);
end
param.Ps = Ps;
param.Rs = Rs;
param.relaxPrecs = relaxPrecs;
param = adjustMemoryForNumRHS(param,MGType,nrhs,verbose);
param.doTranspose = 0;
return param;
end

function adjustMemoryForNumRHS(param::MGparam,MGType::DataType = Float64,nrhs::Int64 = 1,verbose::Bool=false)
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
	memKcycle = Array{FGMRESmem}(undef,max(param.levels-2,0)); 
else
	memKcycle = Array{FGMRESmem}(undef, 0);
end

if param.relaxType == "Jac-GMRES"
	memRelax = Array{FGMRESmem}(undef,param.levels-1); 
else
	memRelax = Array{FGMRESmem}(undef,0);
end

memCycle = Array{CYCLEmem}(undef,param.levels);

param.memRelax  = memRelax;
param.memKcycle = memKcycle;
param.memCycle  = memCycle;

N = size(param.As[1],2); 

for l = 1:(param.levels-1)
	N = size(param.As[l],2);
	needZ = param.relaxType=="Jac-GMRES";
	# In principle, no need for b on the first level.
	memCycle[l] = getCYCLEmem(N,nrhs,MGType,true);
	if param.relaxType=="Jac-GMRES"
		maxRelax = max(param.relaxPre(l),param.relaxPost(l));
		if nrhs == 1
			memRelax[l] = getFGMRESmem(N,true,MGType,maxRelax,1);
		else
			# memRelax[l] = getBlockFGMRESmem(N,nrhs,false,MGType,maxRelax);
			# memRelax[l] = getFGMRESmem(N,false,MGType,maxRelax,nrhs);
			memRelax[l] = getFGMRESmem(N,true,MGType,maxRelax,1);
		end
	end
	
	if l > 1 && param.cycleType=='K'
		if nrhs == 1
			memKcycle[l-1] = getFGMRESmem(N,true,MGType,2,1);
		else
			# memKcycle[l-1] = getBlockFGMRESmem(N,nrhs,true,MGType,2);
			memKcycle[l-1] = getFGMRESmem(N,true,MGType,2,nrhs);
		end
		
	end
end
memCycle[end] = getCYCLEmem(size(param.As[end],2),nrhs,MGType,false); # no need for residual on the coarsest level...
memCycle[end].b = memCycle[end].r;
param.memRelax = memRelax;
param.memKcycle = memKcycle;
param.memCycle = memCycle;
return param;
end


function replaceMatrixInHierarchy(param,AT::SparseMatrixCSC,verbose::Bool=false)
MGType = getMGType(param,eltype(AT));
param.As[1] = convert(SparseMatrixCSC{MGType,spIndType},AT);
Cop = nnz(AT);
T_time = 0;	
for l = 1:(param.levels-1)
	if verbose
		T_time = time_ns();
	end
    AT = param.As[l];
	
	if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES"
		d = param.relaxParam./diag(AT);
		d = spdiagm(d);
		param.relaxPrecs[l] = d;
		if param.singlePrecision
			param.relaxPrecs[l] = convert(SparseMatrixCSC{MGType,spIndType},d);# here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels
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
	if verbose; println("MG setup: took:",(time_ns() - T_time)/1e+9); end;
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

param.As[1] = sparse(param.As[1]');
param.doTranspose = mod(param.doTranspose+1,2);
T_time = 0;
for l = 1:(param.levels-1)
	if verbose
		println("MG transpose level: ",l);
		T_time = time_ns();
	end

    if param.relaxType=="Jac" || param.relaxType=="Jac-GMRES" || param.relaxType=="SPAI"
		param.relaxPrecs[l] = conj(param.relaxPrecs[l]);
	else
		error("Not supported");
	end
	param.Ps[l] = sparse(param.Rs[l]');
	param.Rs[l] = sparse(param.Ps[l]');
	
	param.As[l+1]  = sparse(param.As[l+1]');
end
if verbose 
	T_time = time_ns();
end

if param.coarseSolveType == "MUMPS"
	# when applying mumps, no need to transpose the factorization
elseif param.coarseSolveType == "BiCGSTAB" || param.coarseSolveType == "GMRES"
	param.LU = conj(param.LU);
	
elseif param.coarseSolveType == "VankaFaces"
	error("See about this");
else
	destroyCoarsestLU(param);
	param.LU = lu(sparse(param.As[end]'));
end
if verbose 
	println("MG transpose hierarchy: done LU in ",(time_ns()-T_time)/1e+9 );
	
end
return;
end



export defineCoarsestAinv;
function defineCoarsestAinv(param::MGparam,AT::SparseMatrixCSC)
if isa(param.LU,AbstractSolver)
	println("Using AbstractSolver");
	param.LU = setupSolver(AT,param.LU);
else
	if param.coarseSolveType == "MUMPS"
		param.LU = factorMUMPS(AT',0,0);
	elseif param.coarseSolveType == "BiCGSTAB" || param.coarseSolveType == "GMRES"
		d = sparse(Diagonal(param.relaxParam./diag(AT)));
		param.LU = d;
		if param.singlePrecision
			param.LU = convert(SparseMatrixCSC{spValType,spIndType},d);# here we need to take the conjugate for the SpMatVec, but we give At instead of A so it cancels	
		end
	elseif param.coarseSolveType == "VankaFaces"
		mixedFormulation = false;
		if param.transferOperatorType=="SystemsFacesMixedLinear"
			mixedFormulation = true;
		end	
		Kaczmarz = false;
		innerPrec = 1;
		HKparam = getHybridCellWiseParam(AT,param.Meshes[end],[2,2],0.5,param.numCores,innerPrec,mixedFormulation,Kaczmarz);
		MGType = getMGType(param,eltype(AT));	
		x = zeros(MGType,size(AT,2));
		param.LU = getHybridCellWisePrecond(HKparam,AT,x,mixedFormulation,Kaczmarz);
	else
		param.LU = lu(sparse(AT'));
		# nnzLU = nnz(param.LU[:L]) + nnz(param.LU[:U]);
		# println("Cop coarsest (Julia): ",nnzLU./nnz(param.As[1]));
	end
end
end


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
			warn("getFWInterp: in geometric mode we stop coarsening because num cells does not divide by two");
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


function getSPAIprec(A::SparseMatrixCSC)
s = vec(sum(real(A).^2,dims=2) + sum(imag(A).^2,dims=2));
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




