module Multigrid
using jInv.Mesh;
using jInv.LinearSolvers;
using KrylovMethods

# check if MUMPS can be used
const minMUMPSversion = VersionNumber(0,0,1)
hasMUMPS=false
vMUMPS = VersionNumber(0,0,0)
try 
	vMUMPS = Pkg.installed("MUMPS")
	hasMUMPS = vMUMPS >= minMUMPSversion
	if hasMUMPS
		println("MUMPS!!")
		using MUMPS;
	end
catch 
end

# check if ParSPMatVec is available
hasParSpMatVec = false
const minVerParSpMatVec = VersionNumber(0,0,1)
	vParSpMatVec = VersionNumber(0,0,0)
try 
	vParSpMatVec = Pkg.installed("ParSpMatVec")
	
	hasParSpMatVec = vParSpMatVec>=minVerParSpMatVec
catch 
end
if hasParSpMatVec
	using ParSpMatVec
	# println("USING ParSpMatVec!!!")
end

export MGparam;
export getMGparam,MGsetup, clear!
export BlockFGMRES,ArrayTypes,hierarchyExists,copySolver,destroyCoarsestLU


SparseCSCTypes = Union{SparseMatrixCSC{Complex128,Int64},SparseMatrixCSC{Float64,Int64}}
ArrayTypes = Union{Array{Complex128},Array{Complex64},Array{Float64},Array{Float32}}

include("SpMatMul.jl");
include("FGMRES.jl");
include("BlockFGMRES.jl");

"""
type Multigrid.CYCLEmem
	
Fields:

	b::ArrayTypes - memory for the right-hand-side
	r::ArrayTypes - memory for the residual
	x::ArrayTypes - memory for the iterated solution
"""
type CYCLEmem
	b 					::ArrayTypes
	r					::ArrayTypes
	x					::ArrayTypes
end

"""
type Multigrid.MGparam
	
Fields:

	levels::Int64            - Maximum number of multigrid levels
	numCores::Int64          - Number of OMP cores to work with. Some operations (setup) are not parallelized.
	maxOuterIter::Int64      - Maximum outer iterations.
	relativeTol::Float64	 - Relative L2/Frobenius norm stopping criterion.
	relaxType:: String	 - Relax type. Can be "Jac", "Jac-GMRES" or "SPAI". 
	relaxParam::Float64	     - Relax damping parameter. 
	relaxPre::Function 	     - pre and post relaxation numbers
	relaxPost::Function	     - Can be 'V', 'F', 'W', 'K' (Krylov cycles are done with FGMRES).
	Ps::Array{SparseMatrixCSC{Float64}} - all matrices here are transposed/conjugated so that parallel multiplication is efficient
	Rs::Array{SparseMatrixCSC{Float64}} - all matrices here are transposed/conjugated so that parallel multiplication is efficient
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
""" 
type MGparam
	levels				:: Int64
	numCores			:: Int64
	maxOuterIter		:: Int64
	relativeTol			:: Float64
	relaxType		    :: String
	relaxParam			:: Union{Array{Float64},Float64}
	relaxPre			:: Function
	relaxPost			:: Function
	cycleType			:: Char
	Ps					:: Array{SparseMatrixCSC{Float64}}
	Rs					:: Array{SparseMatrixCSC{Float64}}
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
end

include("MGsetup.jl");
include("SA-AMG.jl");
include("MGcycle.jl");
include("SolveFuncs.jl");
include("SAAMGWrapper.jl");
#include("Systems.jl");

"""
function Multigrid.copySolver(MG::MGparam)

copies the solver parameters without the setup and allocated memory.
"""
function copySolver(MG::MGparam)
return getMGparam(MG.levels,MG.numCores,MG.maxOuterIter,MG.relativeTol,MG.relaxType,MG.relaxParam,
					MG.relaxPre,MG.relaxPost,MG.cycleType,MG.coarseSolveType,MG.strongConnParam,MG.FilteringParam);
end



function getMGparam(levels::Int64,numCores::Int64,maxIter::Int64,relativeTol:: Float64,relaxType::String,relaxParam,
					relaxPre::Function,relaxPost::Function,cycleType::Char='V',coarseSolveType::String="NoMUMPS",strongConnParam::Float64=0.4,FilteringParam::Float64 = 0.0)
return MGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPre,relaxPost,cycleType,[],[],[],[],Array(CYCLEmem,0),
				Array(FGMRESmem,0),Array(FGMRESmem,0),coarseSolveType,[],0,strongConnParam,FilteringParam,Array(RegularMesh,0));
end
					
function getMGparam(levels::Int64=3,numCores::Int64=8,maxIter::Int64=20,relativeTol::Float64=1e-6,relaxType::String="SPAI",relaxParam=1.0,
					relaxPre::Int64=2,relaxPost::Int64=2,cycleType::Char='V',coarseSolveType::String="NoMUMPS",strongConnParam::Float64=0.4,FilteringParam::Float64 = 0.0)
relaxPreFun(x) = relaxPre;
relaxPostFun(x) = relaxPost;
return getMGparam(levels,numCores,maxIter,relativeTol,relaxType,relaxParam,relaxPreFun,relaxPostFun,cycleType,coarseSolveType,strongConnParam,FilteringParam);
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

import jInv.Utils.clear!
function clear!(param::MGparam)
param.Ps = Array(SparseMatrixCSC,0);
param.Rs = Array(SparseMatrixCSC,0);
param.As = Array(SparseMatrixCSC,0);
param.relaxPrecs = [];
param.memCycle = Array(CYCLEmem,0);
param.memRelax = Array(FGMRESmem,0);
param.memKcycle = Array(FGMRESmem,0);
param.Meshes = Array(RegularMesh,0);
destroyCoarsestLU(param);
end

function destroyCoarsestLU(param::MGparam)
param.doTranspose = 0;
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

export defineCoarsestAinv;
function defineCoarsestAinv(param::MGparam,AT::SparseMatrixCSC)
if param.coarseSolveType == "MUMPS"
	param.LU = factorMUMPS(AT',0,0);
elseif param.coarseSolveType == "BiCGSTAB"
	d = param.relaxParam./diag(AT);
	param.LU = spdiagm(d);
else
	param.LU = lufact(AT');
end
param.doTranspose = 0;
end



function hierarchyExists(param::MGparam)
return length(param.As) > 0;
end



end # Module
