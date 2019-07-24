using jInv.Mesh;
using KrylovMethods
using Multigrid.DomainDecomposition

import jInv.LinearSolvers.copySolver;
#import jInv.LinearSolvers.AbstractSolver
#import jInvLinearSolvers.SparseCSCTypes
#import jInvLinearSolvers.ArrayTypes
#import jInvLinearSolvers.hasParSpMatVec

export MGparam;
export getMGparam, MGsetup, clear!
export BlockFGMRES,hierarchyExists,copySolver,destroyCoarsestLU
export multilevelOperatorConstructor, getMultilevelOperatorConstructor


include("SpMatMul.jl");
include("FGMRES.jl");
include("BlockFGMRES.jl");

"""
mutable struct Multigrid.multilevelOperatorConstructor
	
Fields:

	param - parameters for the PDE
	getOperator(mesh,param) - a function that creates the PDE from 
	restrictParams(mesh_fine,mesh_coarse,param_fine,level) - returns param_coarse for level+1.
	
"""
mutable struct multilevelOperatorConstructor
	param			
	getOperator		::Function
	restrictParams  ::Function
end

function getMultilevelOperatorConstructor(param,getOperator::Function,restrictParams)
if restrictParams == []
	## PDE with no params
	restrictParams2(mesh_fine,mesh_coarse,param_fine,level) = [];
	getOperator2(Mesh,param) = getOperator(Mesh);
	return multilevelOperatorConstructor([],getOperator2,restrictParams2);
else
	return multilevelOperatorConstructor(param,getOperator,restrictParams);
end
end
"""
mutable struct Multigrid.CYCLEmem
	
Fields:

	b::ArrayTypes - memory for the right-hand-side
	r::ArrayTypes - memory for the residual
	x::ArrayTypes - memory for the iterated solution
"""
mutable struct CYCLEmem
	b 					::ArrayTypes
	r					::ArrayTypes
	x					::ArrayTypes
end

"""
mutable struct Multigrid.MGparam
	
Fields:

	levels::Int64            - Maximum number of multigrid levels
	numCores::Int64          - Number of OMP cores to work with. Some operations (setup) are not parallelized.
	maxOuterIter::Int64      - Maximum outer iterations.
	relativeTol::Float64	 - Relative L2/Frobenius norm stopping criterion.
	relaxType:: String	 - Relax type. Can be "Jac", "Jac-GMRES" or "SPAI". 
	relaxParam::Float64	     - Relax damping parameter. 
	relaxPre::Function 	     - pre and post relaxation numbers
	relaxPost::Function	     - Can be 'V', 'F', 'W', 'K' (Krylov cycles are done with FGMRES).
	Ps::Array{SparseCSCTypes} - all matrices here are transposed/conjugated so that parallel multiplication is efficient
	Rs::Array{SparseCSCTypes} - all matrices here are transposed/conjugated so that parallel multiplication is efficient
	As::Array{SparseCSCTypes} - all matrices here are transposed/conjugated so that parallel multiplication is efficient
	relaxPrecs -  an array of relaxation preconditioners for all levels.
	memCycle::Array{CYCLEmem} - Space for x,b and r for each level.			
	memRelax::Union{Array{FGMRESmem},Array{BlockFGMRESmem}}  - This is used just in case of GMRES relaxation.
	memKcycle::Union{Array{FGMRESmem},Array{BlockFGMRESmem}} - Memory for the Krylov-cycle FGMRES. First field is ignored.
	coarseSolveType::String - Can be "MUMPS" or "NoMUMPS" for Julia backslash.
	LU - Factorization of coarsest level.
	doTranspose::Int64
	strongConnParam::Float64  - (for SA-AMG only) A threshold for determining a strong connection should >0.25, and <0.85. 
	FilteringParam::Float64	  - (for SA-AMG only) A threshold for prolongation filtering >0.0, and <0.2.
	Meshes::Array{RegularMesh} 		 - Array of Regular Meshes for geometric multigrid.
	transferOperatorType:: String	- (for geometric MG only) May be "FullWeighting", "SystemsFacesLinear"  
""" 
mutable struct MGparam
	levels				:: Int64
	numCores			:: Int64
	maxOuterIter		:: Int64
	relativeTol			:: Float64
	relaxType		    :: String
	relaxParam			:: Union{Array{Float64},Float64}
	relaxPre			:: Function
	relaxPost			:: Function
	cycleType			:: Char
	Ps					:: Array{SparseCSCTypes}
	Rs					:: Array{SparseCSCTypes}
	As					:: Array{SparseCSCTypes}
	relaxPrecs
	memCycle			:: Array{CYCLEmem}
	memRelax			:: Union{Array{FGMRESmem},Array{BlockFGMRESmem}}
	memKcycle			:: Union{Array{FGMRESmem},Array{BlockFGMRESmem}}
	coarseSolveType		:: String
	LU
	doTranspose			:: Int64
	strongConnParam		:: Float64
	FilteringParam		:: Float64
	Meshes				:: Array{RegularMesh}
	transferOperatorType:: String
	singlePrecision		:: Bool
end


const spValType = ComplexF32

include("MGsetup.jl");
include("SA-AMG.jl");
include("MGcycle.jl");
include("SolveFuncs.jl");
include("SAAMGWrapper.jl");
include("Systems.jl");
include("parRelax.jl");
include("Vanka.jl");
include("ClassicalAMG.jl")
"""
function Multigrid.copySolver(MG::MGparam)

copies the solver parameters without the setup and allocated memory.
"""
function copySolver(MG::MGparam)
return getMGparam(MG.levels,MG.numCores,MG.maxOuterIter,MG.relativeTol,MG.relaxType,MG.relaxParam,
					MG.relaxPre,MG.relaxPost,MG.cycleType,MG.coarseSolveType,MG.strongConnParam,MG.FilteringParam,MG.transferOperatorType,MG.singlePrecision);
end



function getMGparam(levels::Int64,numCores::Int64,maxIter::Int64,relativeTol:: Float64,relaxType::String,relaxParam,
					relaxPre::Function,relaxPost::Function,cycleType::Char='V',coarseSolveType::String="NoMUMPS",strongConnParam::Float64=0.4,FilteringParam::Float64 = 0.0,transferOperatorType = "FullWeighting",singlePrecision = false)
return MGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,[],[],[],[],Array{CYCLEmem}(undef,0),
				Array{FGMRESmem}(undef,0),Array{FGMRESmem}(undef,0),coarseSolveType,[],0,strongConnParam,FilteringParam,Array{RegularMesh}(undef,0),transferOperatorType,singlePrecision);
end
					
function getMGparam(levels::Int64=3,numCores::Int64=8,maxIter::Int64=20,relativeTol::Float64=1e-6,relaxType::String="SPAI",relaxParam=1.0,
					relaxPre::Int64=2,relaxPost::Int64=2,cycleType::Char='V',coarseSolveType::String="NoMUMPS",strongConnParam::Float64=0.4,FilteringParam::Float64 = 0.0,transferOperatorType = "FullWeighting",singlePrecision = false)
relaxPreFun(x) = relaxPre;
relaxPostFun(x) = relaxPost;
return getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPreFun,relaxPostFun,cycleType,coarseSolveType,strongConnParam,FilteringParam,transferOperatorType,singlePrecision);
end
					
function getCYCLEmem(n::Int64,m::Int64,T::Type,withB::Bool=true)
b = zeros(T,0);
if m==1
	if withB
		b = zeros(T,n);
	end
	return CYCLEmem(b,zeros(T,n),zeros(T,n));
else
	if withB
		b = zeros(T,n,m);
	end
	return CYCLEmem(b,zeros(T,n,m),zeros(T,n,m));
end
end

# import jInv.Utils.clear!
function clear!(param::MGparam)
param.Ps = Array{SparseMatrixCSC}(undef,0);
param.Rs = Array{SparseMatrixCSC}(undef,0);
param.As = Array{SparseMatrixCSC}(undef,0);
param.relaxPrecs = [];
param.memCycle = Array{CYCLEmem}(undef,0);
param.memRelax = Array{FGMRESmem}(undef,0);
param.memKcycle = Array{FGMRESmem}(undef,0);
param.Meshes = Array{RegularMesh}(undef,0);
destroyCoarsestLU(param);
end

function destroyCoarsestLU(param::MGparam)
if param.LU==[]
	return;
end
if param.coarseSolveType=="MUMPS"
	if isa(param.LU,MUMPSfactorization)
		destroyMUMPS(param.LU);
		param.LU = [];
	end
else
	param.LU = [];
end
return;
end

function getMGType(param::MGparam,btype::DataType)
if btype <: Real
	return param.singlePrecision ? Float32 : Float64;
else
	return param.singlePrecision ? Complex{Float32} : Complex{Float64};
end
end

function getMGType(param::MGparam,b::Array)
if eltype(b) <: Real
	return param.singlePrecision ? Float32 : Float64;
else
	return param.singlePrecision ? Complex{Float32} : Complex{Float64};
end
end

function hierarchyExists(param::MGparam)
return length(param.As) > 0;
end

