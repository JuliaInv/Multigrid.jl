export MGsetup,transposeHierarchy,adjustMemoryForNumRHS,getSPAIprec,replaceMatrixInHierarchy


const spIndType = Int64
export spIndType;

function MGsetup(ATf::Union{SparseMatrixCSC,multilevelOperatorConstructor},Mesh::RegularMesh,param::MGparam{VAL,IND},nrhs::Int64 = 1,verbose::Bool=false) where {VAL,IND}
Ps      	= Array{SparseMatrixCSC}(undef,param.levels-1);
Rs      	= Array{SparseMatrixCSC}(undef,param.levels-1);
As 			= Array{SparseMatrixCSC}(undef,param.levels);
Meshes  	= Array{RegularMesh}(undef,param.levels); 
relaxPrecs 	= Array{Any}(undef,param.levels);

relaxParamArr = zeros(param.levels);

if length(param.relaxParam)==1
	relaxParamArr[:] .= param.relaxParam;
else
	relaxParamArr[:] .= param.relaxParam;
end

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
	geometric = !isa(ATf,SparseMatrixCSC);
	if param.transferOperatorType=="FullWeighting"
		(P,nc) = getFWInterp(n.+1,geometric);
		nc = nc.-1;
		RT = copy(P);
		PT = sparse(P');
		if geometric
			# we need to make sure that the coarse "Galerkin" operator scales like the geometric stencil.
			RT.nzval.*=(0.5^Meshes[l].dim);
		end
		P = 0;
	elseif param.transferOperatorType=="SystemsFacesLinear" || param.transferOperatorType=="SystemsFacesMixedLinear"
		(P,R,nc) = getLinearOperatorsSystemsFaces(n,withCellsBlock);
		PT = sparse(P');
		RT = sparse(R');
		# PT = copy(R);
		# RT = copy(P);
		P = 0;
		R = 0;
		if geometric
			# we need to make sure that the coarse "Galerkin" operator scales like the geometric stencil.
			RT.nzval.*=(0.5^Meshes[l].dim);
		end
	end

	relaxPrecs[l] = getRelaxPrec(AT,VAL,param.relaxType,relaxParamArr[l],Meshes[l],withCellsBlock);
	
	
	if param.singlePrecision
		PT = convert(SparseMatrixCSC{real(VAL),IND},PT);
		RT = convert(SparseMatrixCSC{real(VAL),IND},RT);
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
param = adjustMemoryForNumRHS(param,nrhs,verbose);
param.doTranspose = 0;
return param;
end



function getRelaxPrec(AT,VAL::Type,relaxType,relaxParam=1.0,Mesh_l=[],withCellsBlock=false)
if relaxType=="Jac" || relaxType=="Jac-GMRES"
	d = Vector(conj(relaxParam./diag(AT)));
	return convert(Array{VAL},d);
elseif relaxType=="SPAI"
	return convert(Array{VAL},conj(relaxParam*getSPAIprec(AT))); 
elseif relaxType=="VankaFaces"
	return setupVankaFacesPreconditioner(AT,Mesh_l, relaxParam, withCellsBlock, FULL_VANKA);
elseif relaxType=="EconVankaFaces"
	return setupVankaFacesPreconditioner(AT,Mesh_l, relaxParam, withCellsBlock, ECON_VANKA);
else
	error("Unknown relaxation type !!!!");
end
end





function adjustMemoryForNumRHS(param::MGparam{VAL,IND},nrhs::Int64 = 1,verbose::Bool=false) where {VAL,IND}
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
	memKcycle = Array{FGMRESmem{VAL}}(undef,max(param.levels-2,0)); 
else
	memKcycle = Array{FGMRESmem{VAL}}(undef, 0);
end

if param.relaxType == "Jac-GMRES"
	memRelax = Array{FGMRESmem{VAL}}(undef,param.levels-1); 
else
	memRelax = Array{FGMRESmem{VAL}}(undef,0);
end

memCycle = Array{CYCLEmem{VAL}}(undef,param.levels);

N = size(param.As[1],2); 
for l = 1:(param.levels-1)
	N = size(param.As[l],2);
	needZ = param.relaxType=="Jac-GMRES";
	# In principle, no need for b on the first level.
	memCycle[l] = getCYCLEmem(N,nrhs,VAL,true);
	if param.relaxType=="Jac-GMRES"
		maxRelax = max(param.relaxPre(l),param.relaxPost(l));
		memRelax[l] = getFGMRESmem(N,VAL,maxRelax,nrhs);
	end
	if l > 1 && param.cycleType=='K'
		memKcycle[l-1] = getFGMRESmem(N,VAL,2,nrhs);
	end
end
memCycle[end] = getCYCLEmem(size(param.As[end],2),nrhs,VAL,false); # no need for residual on the coarsest level...
memCycle[end].b = memCycle[end].r;
param.memRelax = memRelax;
param.memKcycle = memKcycle;
param.memCycle = memCycle;
return param;
end


function replaceMatrixInHierarchy(param::MGparam{VAL,IND},AT::SparseMatrixCSC{VAL,IND},verbose::Bool=false) where {VAL,IND}

relaxParamArr = zeros(param.levels);

if length(param.relaxParam)==1
	relaxParamArr[:] .= param.relaxParam;
else
	relaxParamArr[:] .= param.relaxParam;
end

param.As[1] = convert(SparseMatrixCSC{VAL,IND},AT);
Cop = nnz(AT);
T_time = 0;	
for l = 1:(param.levels-1)
	if verbose
		T_time = time_ns();
	end
    AT = param.As[l];
	withCellsBlock = false;
	if param.transferOperatorType=="SystemsFacesMixedLinear"
		withCellsBlock = true;
	end
	Mesh_l = isempty(param.Meshes) ? [] : param.Meshes[l];
	param.relaxPrecs[l] = getRelaxPrec(AT,VAL,param.relaxType,relaxParamArr[l],Mesh_l,withCellsBlock);
	
	Act = param.Ps[l]*AT*param.Rs[l];
	param.As[l+1] = Act;
	Cop = Cop + nnz(Act);
	if verbose; println("MG setup: took:",(time_ns() - T_time)/1e+9); end;
end
if verbose 
	T_time = time_ns();
	println("MG setup: Operator complexity = ",Cop/nnz(param.As[1]));
end

defineCoarsestAinv(param,param.As[end]);
if verbose 
	println("MG setup coarsest ",param.coarseSolveType,": done coarsest in ",(time_ns()-T_time)/1e+9);
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
function defineCoarsestAinv(param::MGparam{VAL,IND},AT::SparseMatrixCSC{VAL,IND}) where {VAL,IND}
if isa(param.LU,AbstractSolver)
	println("Using AbstractSolver");
	param.LU = setupSolver(AT,param.LU);
else
	if param.coarseSolveType == "MUMPS"
		param.LU = factorMUMPS(AT',0,0);
	elseif param.coarseSolveType == "GMRES"
		param.LU = convert(Array{VAL},Vector(conj(param.relaxParam./diag(AT))));
	elseif param.coarseSolveType == "VankaFaces"
		println("This code is not tested!!!");
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



function getSPAIprec(AT::SparseMatrixCSC)
s = vec(sum(real(AT).^2,dims=2) + sum(imag(AT).^2,dims=2));
Q = Vector(conj(diag(AT))./s);
end

